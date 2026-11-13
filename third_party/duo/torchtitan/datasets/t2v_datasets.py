import os
import math
import json
import torch
import random
import numpy as np
import torch.nn.functional as torch_F

from PIL import Image
from einops import rearrange
from typing import Any, Callable, Optional
from dataclasses import dataclass
from lietorch import SE3

from torch.utils.data import IterableDataset
from torch.distributed.checkpoint.stateful import Stateful

from datasets import Dataset
from datasets.distributed import split_dataset_by_node

from transformers import AutoTokenizer

from torchtitan.config import JobConfig
from torchtitan.models.base.vdm_utils import (
    interpolate_camera_poses, 
    compute_relative_poses, 
    get_plucker_embeddings,
)
from torchtitan.tools.logging import logger
from torchtitan.components.dataloader import ParallelAwareDataloader


@dataclass
class TextToVideoDatasetConfig:
    path: str
    loader: Callable
    data_processor: Callable


def _copy_data_for_safe_split(dataset_info: dict, dp_world_size: int = None):
    dataset_length = len(dataset_info["path"])
    padding_size = math.ceil(dataset_length / dp_world_size) * dp_world_size - dataset_length
    # note uniformly sample padding paths from original dataset
    if padding_size > 0:
        padding_idxs = np.linspace(0, dataset_length - 1, padding_size, dtype=int)
        padding_paths = [dataset_info["path"][idx] for idx in padding_idxs]
        dataset_info["path"].extend(padding_paths)
    return dataset_info


def _load_latent_dataset(dataset_path: list, dp_world_size: int):
    dataset_info = {
        "path": [],
    }
    for i in dataset_path:
        with open(i, mode="r", encoding="utf-8") as f:
            latent_info = json.load(f)
            for sub_latent_info in latent_info:
                dataset_info["path"].append(sub_latent_info["path"])
    dataset = Dataset.from_dict(_copy_data_for_safe_split(dataset_info, dp_world_size))
    return dataset


def _latent_processor(
    sample: dict[str, Any], 
    load_latent_1st_frame: bool = False, 
    load_latent_initial_frame: bool = False, 
    load_camera_info: bool = False,
    use_frameswise_camera_info: bool = False,
) -> dict[str, Any]:
    meta_data = torch.load(sample["path"], weights_only=False)

    sample_dict = {}
    sample_dict["pixel_values_emb"] = meta_data["latent"]
    if load_latent_1st_frame:
        if load_latent_initial_frame:
            length = meta_data["latent_initial_frame"].shape[0]
            sample_dict["latent_1st_frame"] = torch.cat(
                [meta_data["latent_initial_frame"], meta_data["latent_1st_frame"][length:]],
                dim=0
            )
        else:
            sample_dict["latent_1st_frame"] = meta_data["latent_1st_frame"]

    sample_dict["text_emb"] = meta_data["text_emb"]

    sample_dict["latent_num_frames"] = meta_data["latent_num_frames"]
    sample_dict["latent_height"] = meta_data["latent_height"]
    sample_dict["latent_width"] = meta_data["latent_width"]

    if load_camera_info:
        # note extract extrinics
        # extr_data: [f, 4, 4]
        c2ws_mat = meta_data.get("c2ws", None)
        if c2ws_mat is None:
            c1 = int(meta_data.get("video_height", 480) // meta_data["latent_height"])
            c2 = int(meta_data.get("video_width", 832) // meta_data["latent_width"])
            num_tokens = int(meta_data["latent_num_frames"] * meta_data["latent_height"] * meta_data["latent_width"])
            c2ws_plucker_embeddings = torch.zeros([num_tokens, 6 * c1 * c2]).float()
        else:
            len_c2ws = len(c2ws_mat)
            # we use interpolate to get camera pose for each latent instead of c2ws_mat[::4]
            c2ws_mat_interp = interpolate_camera_poses(
                src_indices=np.arange(len_c2ws),
                src_rot_mat=c2ws_mat[:, :3, :3],
                src_trans_vec=c2ws_mat[:, :3, 3],
                tgt_indices=np.linspace(0, len_c2ws-1, (len_c2ws - 1) // 4 + 1),
            )
            c2ws_mat_interp = compute_relative_poses(
                c2ws_mat_interp,
                framewise=use_frameswise_camera_info,
                normalize_trans=True,
            ) # [f, 4, 4]
            # note extract intrinsics
            # intr_data: [f, 4], where 4 = [fx, fy, cx, cy]
            if isinstance(meta_data["intrinsics"], torch.Tensor):
                Ks = meta_data["intrinsics"].float()
            else:
                Ks = torch.from_numpy(meta_data["intrinsics"]).float()
            Ks = Ks[0:1].repeat(len(c2ws_mat_interp), 1) # [f, 4]
            # note get plucker embeddings
            c2ws_plucker_embeddings = get_plucker_embeddings(c2ws_mat_interp, Ks, height=meta_data["video_height"], width=meta_data["video_width"])
            c2ws_plucker_embeddings = rearrange(
                c2ws_plucker_embeddings, 
                'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                c1=int(meta_data["video_height"]//meta_data["latent_height"]),
                c2=int(meta_data["video_width"]//meta_data["latent_width"]),
            )
        sample_dict["c2ws_plucker_emb"] = c2ws_plucker_embeddings # [f*h*w, c]
    return sample_dict


DATASETS = {
    "latent": TextToVideoDatasetConfig(
        path=["/ssd0/zelin/data/meta_latent_info.json"],
        loader=lambda path, dp_world_size: _load_latent_dataset(path, dp_world_size),
        data_processor=_latent_processor,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: Optional[str] = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.data_processor


class LatentDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        job_config: Optional[JobConfig] = None,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        is_val: bool = False,
    ) -> None:

        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path_list, dataset_loader, data_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path_list, dp_world_size)
        
        self.is_val = is_val
        self.dp_rank = dp_rank
        self.dataset_name = dataset_name
        
        if job_config.training.split_train_data:
            if not is_val:
                logger.info(f"split train dataset")
                self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
            else:
                self._data = ds
        else:
            self._data = ds
        self.empty_text_emb = torch.load(job_config.vdm.empty_text_emb_path, weights_only=False)

        self._data_processor = data_processor
        self.job_config = job_config

        self.infinite = infinite

        # Variables for checkpointing
        self._epoch = 0
        self._sample_idx = 0
        self._all_samples: list[dict[str, Any]] = []
        logger.info(f"Dataset {self.dataset_name} epoch {self._epoch} starting.")

    def _get_data_iter(self, shuffle=True):
        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        it = iter(self._data.shuffle(seed=self.dp_rank) if shuffle else self._data)
        for _ in range(self._sample_idx):
            next(it)
        return it

    def __iter__(self):
        max_buffer_token_len = self.job_config.training.seq_len
        dataset_iterator = self._get_data_iter(shuffle=True if not self.is_val else False)
        # note load initial frames (i2v or v2v)
        load_latent_1st_frame = self.job_config.vdm.generation_mode == "i2v"
        # note load camera info
        load_camera_info = self.job_config.vdm.load_camera_info
        while True:
            buffer_dict = {
                "pixel_values_emb": [],
                "text_emb": [],
                "text_emb_neg": [],
                "text_emb_len_lst": [],
                "latent_num_frames_lst": [],
                "latent_height_lst": [],
                "latent_width_lst": [],
                "latent_len_lst": [],
                "n_seq": 0,
            }
            if load_latent_1st_frame:
                buffer_dict["latent_1st_frame"] = []
                buffer_dict["load_latent_initial_frame"] = []
            if load_camera_info:
                buffer_dict["c2ws_emb"] = []
                buffer_dict["c2ws_plucker_emb"] = []

            cur_buffer_token_len = 0
            while cur_buffer_token_len < max_buffer_token_len:
                if self.job_config.training.single_batch and buffer_dict["n_seq"] == 1:
                    # note force batch size = 1, avoid fetching & processing a wasted sample
                    break
                try:
                    sample = next(dataset_iterator)
                except StopIteration:
                    if not self.infinite:
                        # return all buffer data
                        if buffer_dict["n_seq"] > 0:
                            break
                        else:
                            return
                    else:
                        self._epoch += 1
                        self._sample_idx = 0
                        logger.info(f"Dataset {self.dataset_name} epoch {self._epoch} starting.")
                        # restart the loop
                        dataset_iterator = self._get_data_iter(shuffle=True if not self.is_val else False)
                        sample = next(dataset_iterator)
                
                # Use the dataset-specific preprocessor
                try:
                    load_latent_initial_frame = (
                        load_latent_1st_frame and 
                        self.job_config.vdm.initial_num_frames > 1 and 
                        random.random() < self.job_config.vdm.load_latent_initial_frame_prob
                    )
                    use_frameswise_camera_info = self.job_config.vdm.use_frameswise_camera_info
                    sample_dict = self._data_processor(
                        sample, 
                        load_latent_1st_frame, 
                        load_latent_initial_frame,
                        load_camera_info,
                        use_frameswise_camera_info,
                    )
                except Exception as e:
                    logger.error(f"Error processing sample {sample['path']}: {e}")
                    self._sample_idx += 1
                    continue

                # Filter out very short videos (minimum 21 latent frames = 81 video frames)
                if sample_dict["latent_num_frames"] < 21:
                    self._sample_idx += 1
                    continue

                # Exceed max_buffer_token_len, break the loop
                sample_token_len = sample_dict["pixel_values_emb"].shape[0]
                if cur_buffer_token_len + sample_token_len > max_buffer_token_len:
                    if sample_token_len > max_buffer_token_len:
                        raise RuntimeError(f"Single sample token {sample_token_len} exceeds max_buffer_token_len {max_buffer_token_len}")
                    else:
                        break

                # Load data into buffer dict
                buffer_dict["pixel_values_emb"].append(sample_dict["pixel_values_emb"])
                if random.random() < self.job_config.training.classifer_free_guidance_prob and not self.is_val: # CFG
                    buffer_dict["text_emb"].append(self.empty_text_emb.clone())
                else:
                    buffer_dict["text_emb"].append(sample_dict["text_emb"])
                buffer_dict["text_emb_neg"].append(self.empty_text_emb.clone())
                buffer_dict["text_emb_len_lst"].append(sample_dict["text_emb"].shape[0])
                buffer_dict["latent_num_frames_lst"].append(sample_dict["latent_num_frames"])
                buffer_dict["latent_height_lst"].append(sample_dict["latent_height"])
                buffer_dict["latent_width_lst"].append(sample_dict["latent_width"])
                buffer_dict["latent_len_lst"].append(sample_dict["latent_num_frames"]*sample_dict["latent_height"]*sample_dict["latent_width"])
                buffer_dict["n_seq"] += 1
                if load_latent_1st_frame:
                    latent_1st_frame = sample_dict["latent_1st_frame"]
                    if latent_1st_frame.shape[0] != sample_dict["pixel_values_emb"].shape[0]:
                        raise RuntimeError(f"latent_1st_frame does not match pixel_values_emb!!")
                    buffer_dict["latent_1st_frame"].append(latent_1st_frame)
                    buffer_dict["load_latent_initial_frame"].append(load_latent_initial_frame)
                if load_camera_info:
                    buffer_dict["c2ws_plucker_emb"].append(sample_dict["c2ws_plucker_emb"])

                cur_buffer_token_len += sample_token_len
                self._sample_idx += 1

            # yield final data_dict
            data_dict = {}
            data_dict["pixel_values_emb"] = torch.cat(buffer_dict["pixel_values_emb"], dim=0)
            data_dict["text_emb"] = torch.cat(buffer_dict["text_emb"], dim=0)
            data_dict["text_emb_neg"] = torch.cat(buffer_dict["text_emb_neg"], dim=0)
            data_dict["text_emb_len_lst"] = buffer_dict["text_emb_len_lst"]
            data_dict["latent_num_frames_lst"] = buffer_dict["latent_num_frames_lst"]
            data_dict["latent_height_lst"] = buffer_dict["latent_height_lst"]
            data_dict["latent_width_lst"] = buffer_dict["latent_width_lst"]
            data_dict["latent_len_lst"] = buffer_dict["latent_len_lst"]
            data_dict["n_seq"] = buffer_dict["n_seq"]
            if load_latent_1st_frame:
                data_dict["latent_1st_frame"] = torch.cat(buffer_dict["latent_1st_frame"], dim=0)
                data_dict["load_latent_initial_frame"] = buffer_dict["load_latent_initial_frame"]
            if load_camera_info:
                data_dict["c2ws_plucker_emb"] = torch.cat(buffer_dict["c2ws_plucker_emb"])
            
            labels = data_dict.pop("pixel_values_emb")
            yield data_dict, labels

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._epoch = state_dict.get("epoch", 0)

    def state_dict(self):
        return {
            "sample_idx": self._sample_idx,
            "epoch": self._epoch,
        }


def build_t2v_dataloader(
    dp_world_size: int,
    dp_rank: int,
    job_config: JobConfig,
    # This parameter is not used, keep it for compatibility
    tokenizer: AutoTokenizer | None,
    infinite: bool = True,
    mode: str = "train",
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size

    if job_config.training.dataset == "latent" or job_config.training.dataset == "latent_sever":
        if mode == "train":
            ds = LatentDataset(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                job_config=job_config,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                infinite=True,
                is_val=False,
            )
        elif mode == "val":
            dataset_path = job_config.training.val_dataset_path
            ds = LatentDataset(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                job_config=job_config,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                infinite=False,
                is_val=True,
            )

    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        num_workers=job_config.training.num_data_workers,
        prefetch_factor=job_config.training.data_prefetch_factor,
        pin_memory=True,
    )