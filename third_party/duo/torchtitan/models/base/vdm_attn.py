import torch
import torch.nn.functional as torch_F

from einops import rearrange
from typing_extensions import List, Callable, ClassVar

from flash_attn_interface import flash_attn_func, flash_attn_varlen_func

from sageattention import sageattn, sageattn_varlen

from magi_attention.common.ranges import AttnRanges
from magi_attention.functional.flex_flash_attn import flex_flash_attn_func


class BaseAttnVarlen(torch.nn.Module):
    def __init__(
        self, 
        attn_type: str = "self-attention",
    ) -> None:
        super().__init__()
        if attn_type not in ["self-attention", "cross-attention"]:
            raise ValueError(f"Unrecognized attn_type {attn_type}.")
        self.attn_type = attn_type

    def convert_type(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        q_varlen = query[0]
        k_varlen = key[0]
        v_varlen = value[0]

        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)
        
        q_varlen = half(q_varlen)
        k_varlen = half(k_varlen)
        v_varlen = half(v_varlen)
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)
        return q_varlen, k_varlen, v_varlen

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype=torch.bfloat16,
        block_idx: int = 0,
    ):
        q_varlen, k_varlen, v_varlen = self.convert_type(query, key, value, dtype)
        if self.attn_type == "self-attention":
            x_out = self.self_attn(q_varlen, k_varlen, v_varlen, block_idx=block_idx)
        elif self.attn_type == "cross-attention":
            x_out = self.cross_attn(q_varlen, k_varlen, v_varlen, block_idx=block_idx)
        else:
            raise ValueError(f"Unrecognized attn_type {self.attn_type}.")
        return x_out

    def self_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_idx: int = 0,
    ):
        raise NotImplementedError("Not implement self.self_attn")

    def cross_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_idx: int = 0,
    ):
        raise NotImplementedError("Not implement self.cross_attn")

    @staticmethod
    @torch.no_grad()
    def init_attn_params(
        self,
    ) -> None:
        raise NotImplementedError("Not implement self.init_attn_params")


class FlexFlashAttnVarlen(BaseAttnVarlen):
    """Magi execution with optional forward-local ranges.

    Explicit ranges never consult or update class-level Wan mask state. This
    lets checkpoint recomputation and interleaved models retain their own mask.
    Legacy callers can continue to initialize the class-level mask as before.
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype=torch.bfloat16,
        block_idx: int = 0,
        *,
        q_ranges: torch.Tensor | None = None,
        k_ranges: torch.Tensor | None = None,
        attn_type_map: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        supplied = (q_ranges is not None, k_ranges is not None, attn_type_map is not None)
        if any(supplied) and not all(supplied):
            raise ValueError('Explicit attention requires q_ranges, k_ranges and attn_type_map together')
        if all(supplied) and any(t.ndim != 4 or t.shape[0] != 1 for t in (query, key, value)):
            raise ValueError('Explicit varlen attention expects one packed sequence [1, tokens, heads, dim]')
        q, k, v = self.convert_type(query, key, value, dtype)
        if all(supplied):
            if not q.is_cuda or torch.cuda.get_device_capability(q.device)[0] < 9:
                raise ValueError('Explicit range masks require Magi on a supported CUDA device; use SDPA for reference tests')
            if (q_ranges.ndim != 2 or q_ranges.shape[1] != 2
                    or k_ranges.shape != q_ranges.shape
                    or attn_type_map.shape != (q_ranges.shape[0],)):
                raise ValueError('Invalid explicit attention range shapes')
            if any(t.dtype != torch.int32 or t.device != q.device
                   for t in (q_ranges, k_ranges, attn_type_map)):
                raise ValueError('Attention ranges/types must be int32 tensors on the query device')
            return self._magi(q, k, v, q_ranges, k_ranges, attn_type_map, deterministic)[None, ...]
        if self.attn_type == 'self-attention':
            return self.self_attn(q, k, v, block_idx=block_idx, deterministic=deterministic)
        return self.cross_attn(q, k, v, block_idx=block_idx, deterministic=deterministic)

    @staticmethod
    def _magi(query, key, value, q_ranges, k_ranges, attn_type_map, deterministic):
        output, _ = flex_flash_attn_func(
            query, key, value, q_ranges=q_ranges, k_ranges=k_ranges,
            attn_type_map=attn_type_map, deterministic=deterministic,
        )
        return output

    # flash attn params
    max_seqlen_q: int = None
    cu_seqlens_q: torch.Tensor = None
    max_seqlen_kv: int = None
    cu_seqlens_kv: torch.Tensor = None
    # magi attn params
    ## self-attention
    selfattn_q_ranges: AttnRanges = None
    selfattn_kv_ranges: AttnRanges = None
    selfattn_mask_type: List[int] = None
    ## cross-attention
    crossattn_q_ranges: AttnRanges = None
    crossattn_kv_ranges: AttnRanges = None
    crossattn_mask_type: List[int] = None
    # cached GPU tensors for to_tensor() results (avoid re-creating per attention layer)
    selfattn_q_ranges_tensor: torch.Tensor = None
    selfattn_kv_ranges_tensor: torch.Tensor = None
    crossattn_q_ranges_tensor: torch.Tensor = None
    crossattn_kv_ranges_tensor: torch.Tensor = None
    # sparse teacher forcing
    selfattn_q_ranges_tensor_lst: ClassVar[List[torch.Tensor]] = None
    selfattn_kv_ranges_tensor_lst: ClassVar[List[torch.Tensor]] = None
    selfattn_mask_type_lst: ClassVar[List[torch.Tensor]] = None
    is_sparse_mask: bool = False
    sparse_tf_last_n_clean_chunks: int = 0

    def self_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_idx: int = 0,
        *,
        deterministic: bool = False,
    ):
        if torch.cuda.get_device_capability()[0] >= 9: # magiattn only support hopper gpus
            if FlexFlashAttnVarlen.is_sparse_mask:
                q_ranges_t = FlexFlashAttnVarlen.selfattn_q_ranges_tensor_lst[block_idx]
                k_ranges_t = FlexFlashAttnVarlen.selfattn_kv_ranges_tensor_lst[block_idx]
                mask_type_t = FlexFlashAttnVarlen.selfattn_mask_type_lst[block_idx]
            else:
                q_ranges_t = FlexFlashAttnVarlen.selfattn_q_ranges_tensor
                k_ranges_t = FlexFlashAttnVarlen.selfattn_kv_ranges_tensor
                mask_type_t = FlexFlashAttnVarlen.selfattn_mask_type
            x_out = self._magi(
                query, key, value, q_ranges_t, k_ranges_t, mask_type_t, deterministic,
            )
        else: # otherwise we use flash attn varlen
            x_out = flash_attn_varlen_func(
                query, key, value,
                cu_seqlens_q=FlexFlashAttnVarlen.cu_seqlens_q,
                cu_seqlens_k=FlexFlashAttnVarlen.cu_seqlens_q,
                max_seqlen_q=FlexFlashAttnVarlen.max_seqlen_q,
                max_seqlen_k=FlexFlashAttnVarlen.max_seqlen_q,
                deterministic=deterministic,
            )
        x_out = x_out[None, ...] # add batch_size dim
        return x_out

    def cross_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_idx: int = 0,
        *,
        deterministic: bool = False,
    ):
        if torch.cuda.get_device_capability()[0] >= 9: # magiattn only support hopper gpus
            x_out = self._magi(
                query, key, value,
                FlexFlashAttnVarlen.crossattn_q_ranges_tensor,
                FlexFlashAttnVarlen.crossattn_kv_ranges_tensor,
                FlexFlashAttnVarlen.crossattn_mask_type,
                deterministic,
            )
        else: # otherwise we use flash attn varlen
            x_out = flash_attn_varlen_func(
                query, key, value,
                cu_seqlens_q=FlexFlashAttnVarlen.cu_seqlens_q,
                cu_seqlens_k=FlexFlashAttnVarlen.cu_seqlens_kv,
                max_seqlen_q=FlexFlashAttnVarlen.max_seqlen_q,
                max_seqlen_k=FlexFlashAttnVarlen.max_seqlen_kv,
                deterministic=deterministic,
            )
        x_out = x_out[None, ...] # add batch_size dim
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_attn_params(
        n_seq: int = None,
        q_seq_len_list: List = None,
        kv_seq_len_list: List = None,
        selfattn_mask_method: str = "cusum",
        latent_num_frames_list: List = None,
        latent_len_per_frame_list: List = None,
        latent_num_frames_per_chunk_list: List = None,
        torch_device: str = "cuda",
        num_blocks: int = None,
        sparse_tf_start: int = 15,
        sparse_tf_end: int = -1,
        sparse_tf_last_n_clean_chunks: int = 1,
    ) -> None:
        if torch.cuda.get_device_capability()[0] >= 9: # magiattn only support hopper gpus
            if selfattn_mask_method == "sparse_teacher_forcing":
                # full mask = current teacher_forcing full-causal TF
                full_q, full_kv = FlexFlashAttnVarlen.compute_ranges(
                    q_seq_len_list, q_seq_len_list,
                    method="teacher_forcing",
                    latent_num_frames_list=latent_num_frames_list,
                    latent_len_per_frame_list=latent_len_per_frame_list,
                    latent_num_frames_per_chunk_list=latent_num_frames_per_chunk_list,
                )
                # sparse mask = frame-diagonal + first frame anchor + last n frame anchor
                FlexFlashAttnVarlen.sparse_tf_last_n_clean_chunks = int(sparse_tf_last_n_clean_chunks)
                sparse_q, sparse_kv = FlexFlashAttnVarlen.compute_ranges(
                    q_seq_len_list, q_seq_len_list,
                    method="teacher_forcing_sparse",
                    latent_num_frames_list=latent_num_frames_list,
                    latent_len_per_frame_list=latent_len_per_frame_list,
                    latent_num_frames_per_chunk_list=latent_num_frames_per_chunk_list,
                )
                full_ranges_q = AttnRanges.from_ranges(full_q)
                full_ranges_kv = AttnRanges.from_ranges(full_kv)
                sparse_ranges_q = AttnRanges.from_ranges(sparse_q)
                sparse_ranges_kv = AttnRanges.from_ranges(sparse_kv)
                full_q_t = full_ranges_q.to_tensor(torch_device)
                full_kv_t = full_ranges_kv.to_tensor(torch_device)
                sparse_q_t = sparse_ranges_q.to_tensor(torch_device)
                sparse_kv_t = sparse_ranges_kv.to_tensor(torch_device)
                full_mask_type = torch.zeros(len(full_ranges_q), dtype=torch.int32, device=torch_device)
                sparse_mask_type = torch.zeros(len(sparse_ranges_q), dtype=torch.int32, device=torch_device)
                # build per-block list
                q_lst, kv_lst, type_lst = [], [], []
                for block_idx in range(num_blocks):
                    if sparse_tf_start <= block_idx < sparse_tf_end:
                        q_lst.append(sparse_q_t)
                        kv_lst.append(sparse_kv_t)
                        type_lst.append(sparse_mask_type)
                    else:
                        q_lst.append(full_q_t)
                        kv_lst.append(full_kv_t)
                        type_lst.append(full_mask_type)
                FlexFlashAttnVarlen.is_sparse_mask = True
                FlexFlashAttnVarlen.selfattn_q_ranges_tensor_lst = q_lst
                FlexFlashAttnVarlen.selfattn_kv_ranges_tensor_lst = kv_lst
                FlexFlashAttnVarlen.selfattn_mask_type_lst = type_lst
            else:
                FlexFlashAttnVarlen.is_sparse_mask = False
                selfattn_q_ranges, selfattn_kv_ranges = FlexFlashAttnVarlen.compute_ranges(
                    q_seq_len_list,
                    q_seq_len_list,
                    method=selfattn_mask_method,
                    latent_num_frames_list=latent_num_frames_list,
                    latent_len_per_frame_list=latent_len_per_frame_list,
                    latent_num_frames_per_chunk_list=latent_num_frames_per_chunk_list,
                )
                selfattn_q_ranges = AttnRanges.from_ranges(selfattn_q_ranges)
                selfattn_kv_ranges = AttnRanges.from_ranges(selfattn_kv_ranges)
                selfattn_mask_type = torch.zeros(len(selfattn_q_ranges), dtype=torch.int32, device=torch_device)
                FlexFlashAttnVarlen.selfattn_q_ranges = selfattn_q_ranges
                FlexFlashAttnVarlen.selfattn_kv_ranges = selfattn_kv_ranges
                FlexFlashAttnVarlen.selfattn_mask_type = selfattn_mask_type
                FlexFlashAttnVarlen.selfattn_q_ranges_tensor = selfattn_q_ranges.to_tensor(torch_device)
                FlexFlashAttnVarlen.selfattn_kv_ranges_tensor = selfattn_kv_ranges.to_tensor(torch_device)
            # cross attention
            crossattn_q_ranges, crossattn_kv_ranges = FlexFlashAttnVarlen.compute_ranges(
                q_seq_len_list,
                kv_seq_len_list,
                method="cusum",
            )
            crossattn_q_ranges = AttnRanges.from_ranges(crossattn_q_ranges)
            crossattn_kv_ranges = AttnRanges.from_ranges(crossattn_kv_ranges)
            crossattn_mask_type = torch.zeros(len(crossattn_q_ranges), dtype=torch.int32, device=torch_device)
            FlexFlashAttnVarlen.crossattn_q_ranges = crossattn_q_ranges
            FlexFlashAttnVarlen.crossattn_kv_ranges = crossattn_kv_ranges
            FlexFlashAttnVarlen.crossattn_mask_type = crossattn_mask_type
            FlexFlashAttnVarlen.crossattn_q_ranges_tensor = crossattn_q_ranges.to_tensor(torch_device)
            FlexFlashAttnVarlen.crossattn_kv_ranges_tensor = crossattn_kv_ranges.to_tensor(torch_device)
        else: # otherwise we use flash attn varlen
            seqlen_q = torch.tensor(q_seq_len_list, dtype=torch.int32, device=torch_device)
            cu_seqlens_q = torch.zeros(n_seq + 1, dtype=torch.int32, device=torch_device)
            cu_seqlens_q[1:] = torch.cumsum(seqlen_q, dim=0)
            FlexFlashAttnVarlen.max_seqlen_q = max(seqlen_q)
            FlexFlashAttnVarlen.cu_seqlens_q = cu_seqlens_q

            seqlen_kv = torch.tensor(kv_seq_len_list, dtype=torch.int32, device=torch_device)
            cu_seqlens_kv = torch.zeros(n_seq + 1, dtype=torch.int32, device=torch_device)
            cu_seqlens_kv[1:] = torch.cumsum(seqlen_kv, dim=0)
            FlexFlashAttnVarlen.max_seqlen_kv = max(seqlen_kv)
            FlexFlashAttnVarlen.cu_seqlens_kv = cu_seqlens_kv

    @staticmethod
    def compute_ranges(
        q_seq_len_list: List, 
        kv_seq_len_list: List, 
        method: str = "cusum",
        latent_num_frames_list: List = None,
        latent_len_per_frame_list: List = None,
        latent_num_frames_per_chunk_list: List = None,
    ):
        q_range_lst = []
        kv_ranges_lst = []
        # note for bidirectional self-attn / cross-attn
        if method == "cusum" or method == "bidirectional":
            q_start = 0
            kv_start = 0
            for (
                q_seq_len, kv_seq_len,
            ) in zip(q_seq_len_list, kv_seq_len_list):
                q_seq_len = int(q_seq_len)
                kv_seq_len = int(kv_seq_len)
                q_range_lst.append([q_start, q_start + q_seq_len])
                kv_ranges_lst.append([kv_start, kv_start + kv_seq_len])
                q_start += q_seq_len
                kv_start += kv_seq_len
        # note for causal self-attn
        elif method == "diffusion_forcing":
            current_start_frame = 0
            for (
                latent_len_per_frame, latent_num_frames, latent_num_frames_per_chunk,
            ) in zip(latent_len_per_frame_list, latent_num_frames_list, latent_num_frames_per_chunk_list):
                latent_num_frames = int(latent_num_frames)
                latent_len_per_frame = int(latent_len_per_frame)
                latent_len_total_frame = latent_num_frames * latent_len_per_frame
                # split by chunk
                chunk_size = int(latent_num_frames_per_chunk)
                num_chunk = (latent_num_frames + chunk_size - 1) // chunk_size
                for chunk_idx in range(num_chunk):
                    chunk_start_frame = chunk_idx * chunk_size
                    chunk_end_frame = min((chunk_idx + 1) * chunk_size, latent_num_frames)
                    q_range_lst.append(
                        [
                            current_start_frame + chunk_start_frame * latent_len_per_frame,
                            current_start_frame + chunk_end_frame * latent_len_per_frame,
                        ]
                    )
                    kv_ranges_lst.append(
                        [
                            current_start_frame + 0,
                            current_start_frame + chunk_end_frame * latent_len_per_frame,
                        ]
                    )
                
                current_start_frame = current_start_frame + latent_len_total_frame
        elif method == "teacher_forcing":
            current_start_frame = 0
            for (
                latent_len_per_frame, latent_num_frames, latent_num_frames_per_chunk,
            ) in zip(latent_len_per_frame_list, latent_num_frames_list, latent_num_frames_per_chunk_list):
                latent_num_frames = int(latent_num_frames)
                latent_len_per_frame = int(latent_len_per_frame)
                latent_len_total_frame = latent_num_frames * latent_len_per_frame
                # note for teacher forcing mask, pls refer to: https://arxiv.org/pdf/2512.15702
                current_start_frame_clean = current_start_frame
                current_start_frame_noisy = current_start_frame + latent_len_total_frame
                # split by chunk
                chunk_size = int(latent_num_frames_per_chunk)
                num_chunk = (latent_num_frames + chunk_size - 1) // chunk_size
                # 1st mask: clean by clean (upper left)
                for chunk_idx in range(num_chunk):
                    chunk_start_frame = chunk_idx * chunk_size
                    chunk_end_frame = min((chunk_idx + 1) * chunk_size, latent_num_frames)
                    q_range_lst.append(
                        [
                            current_start_frame_clean + chunk_start_frame * latent_len_per_frame,
                            current_start_frame_clean + chunk_end_frame * latent_len_per_frame,
                        ]
                    )
                    kv_ranges_lst.append(
                        [
                            current_start_frame_clean + 0,
                            current_start_frame_clean + chunk_end_frame * latent_len_per_frame,
                        ]
                    )
                # 2nd mask: clean by noise (upper right), do nothing
                # todo need more discussions
                # 3rd mask: noise by clean (lower left)
                for chunk_idx in range(1, num_chunk):
                    chunk_start_frame = chunk_idx * chunk_size
                    chunk_end_frame = min((chunk_idx + 1) * chunk_size, latent_num_frames)
                    q_range_lst.append(
                        [
                            current_start_frame_noisy + chunk_start_frame * latent_len_per_frame,
                            current_start_frame_noisy + chunk_end_frame * latent_len_per_frame,
                        ]
                    )
                    kv_ranges_lst.append(
                        [
                            current_start_frame_clean + 0,
                            current_start_frame_clean + chunk_start_frame * latent_len_per_frame,
                        ]
                    )
                # 4th mask: noise by noise (lower right)
                for chunk_idx in range(num_chunk):
                    chunk_start_frame = chunk_idx * chunk_size
                    chunk_end_frame = min((chunk_idx + 1) * chunk_size, latent_num_frames)
                    q_range_lst.append(
                        [
                            current_start_frame_noisy + chunk_start_frame * latent_len_per_frame,
                            current_start_frame_noisy + chunk_end_frame * latent_len_per_frame,
                        ]
                    )
                    kv_ranges_lst.append(
                        [
                            current_start_frame_noisy + chunk_start_frame * latent_len_per_frame,
                            current_start_frame_noisy + chunk_end_frame * latent_len_per_frame,
                        ]
                    )
                # update current_start_frame (2 denote clean & noisy latents)
                current_start_frame = current_start_frame + (2 * latent_len_total_frame)
        elif method == "teacher_forcing_sparse":
            current_start_frame = 0
            last_n_clean_chunks = int(FlexFlashAttnVarlen.sparse_tf_last_n_clean_chunks)
            for (
                latent_len_per_frame, latent_num_frames, latent_num_frames_per_chunk,
            ) in zip(latent_len_per_frame_list, latent_num_frames_list, latent_num_frames_per_chunk_list):
                latent_num_frames = int(latent_num_frames)
                latent_len_per_frame = int(latent_len_per_frame)
                latent_len_total_frame = latent_num_frames * latent_len_per_frame
                current_start_frame_clean = current_start_frame
                current_start_frame_noisy = current_start_frame + latent_len_total_frame
                chunk_size = int(latent_num_frames_per_chunk)
                num_chunk = (latent_num_frames + chunk_size - 1) // chunk_size
                first_frame_kv_end = min(chunk_size, latent_num_frames) * latent_len_per_frame
                # note self chunk anchor
                for chunk_idx in range(num_chunk):
                    chunk_start_frame = chunk_idx * chunk_size
                    chunk_end_frame = min((chunk_idx + 1) * chunk_size, latent_num_frames)
                    # clean by clean (upper left)
                    q_range_lst.append(
                        [
                            current_start_frame_clean + chunk_start_frame * latent_len_per_frame, 
                            current_start_frame_clean + chunk_end_frame * latent_len_per_frame, 
                        ]
                    )
                    kv_ranges_lst.append(
                        [
                            current_start_frame_clean + chunk_start_frame * latent_len_per_frame, 
                            current_start_frame_clean + chunk_end_frame * latent_len_per_frame, 
                        ]
                    )
                    # noisy by noisy (lower right)
                    q_range_lst.append(
                        [
                            current_start_frame_noisy + chunk_start_frame * latent_len_per_frame, 
                            current_start_frame_noisy + chunk_end_frame * latent_len_per_frame, 
                        ]
                    )
                    kv_ranges_lst.append(
                        [
                            current_start_frame_noisy + chunk_start_frame * latent_len_per_frame, 
                            current_start_frame_noisy + chunk_end_frame * latent_len_per_frame, 
                        ]
                    )
                # note first chunk anchor (skip chunk 0)
                for chunk_idx in range(1, num_chunk):
                    chunk_start_frame = chunk_idx * chunk_size
                    chunk_end_frame = min((chunk_idx + 1) * chunk_size, latent_num_frames)
                    # clean by clean (upper left)
                    q_range_lst.append(
                        [
                            current_start_frame_clean + chunk_start_frame * latent_len_per_frame, 
                            current_start_frame_clean + chunk_end_frame * latent_len_per_frame, 
                        ]
                    )
                    kv_ranges_lst.append(
                        [
                            current_start_frame_clean, 
                            current_start_frame_clean + first_frame_kv_end, 
                        ]
                    )
                    # noisy by clean (lower right)
                    q_range_lst.append(
                        [
                            current_start_frame_noisy + chunk_start_frame * latent_len_per_frame, 
                            current_start_frame_noisy + chunk_end_frame * latent_len_per_frame, 
                        ]
                    )
                    kv_ranges_lst.append(
                        [
                            current_start_frame_clean,
                            current_start_frame_clean + first_frame_kv_end,
                        ]
                    )
                # note last n chunk anchor
                if last_n_clean_chunks > 0:
                    for chunk_idx in range(2, num_chunk):
                        # here is the example for better understanding
                        # last_n_clean_chunks: 3
                        # chunk idx: 2
                        # last_kv_start_chunk : max(1,2-3) = 1
                        # last_kv_start-end: [1, 2) * chunk_size => chunk 1
                        # chunk idx: 3
                        # last_kv_start_chunk : max(1,3-3) = 1
                        # last_kv_start-end: [1, 3) * chunk_size => chunk 1,2
                        # chunk idx: 4
                        # last_kv_start_chunk : max(1,4-3) = 1
                        # last_kv_start-end: [1, 4) * chunk_size => chunk 1,2,3
                        # chunk idx: 5
                        # last_kv_start_chunk : max(1,5-3) = 2
                        # last_kv_start-end: [2, 5) * chunk_size => chunk 2,3,4
                        last_kv_start_chunk = max(1, chunk_idx - last_n_clean_chunks)
                        last_kv_start_frame = last_kv_start_chunk * chunk_size
                        last_kv_end_frame = chunk_idx * chunk_size
                        chunk_start_frame = chunk_idx * chunk_size
                        chunk_end_frame = min((chunk_idx + 1) * chunk_size, latent_num_frames)
                        # clean by clean (upper left)
                        q_range_lst.append(
                            [
                                current_start_frame_clean + chunk_start_frame * latent_len_per_frame,
                                current_start_frame_clean + chunk_end_frame * latent_len_per_frame,
                            ]
                        )
                        kv_ranges_lst.append(
                            [
                                current_start_frame_clean + last_kv_start_frame * latent_len_per_frame,
                                current_start_frame_clean + last_kv_end_frame * latent_len_per_frame,
                            ]
                        )
                        # noisy by clean (lower left)
                        q_range_lst.append(
                            [
                                current_start_frame_noisy + chunk_start_frame * latent_len_per_frame,
                                current_start_frame_noisy + chunk_end_frame * latent_len_per_frame,
                            ]
                        )
                        kv_ranges_lst.append(
                            [
                                current_start_frame_clean + last_kv_start_frame * latent_len_per_frame,
                                current_start_frame_clean + last_kv_end_frame * latent_len_per_frame,
                            ]
                        )
                # note update current_start_frame (2 denote clean & noisy latents)
                current_start_frame = current_start_frame + (2 * latent_len_total_frame)
        else:
            raise ValueError(f"Unsupported copmute ranges method {method}")
        return q_range_lst, kv_ranges_lst


class FlexFlashAttnVarlen_Simple(BaseAttnVarlen):
    def self_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_idx: int = 0,
    ):
        # todo hardcode batch size = 1
        x_out = flash_attn_func(query[None, ...], key[None, ...], value[None, ...]) # add batch_size dim
        return x_out

    def cross_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_idx: int = 0,
    ):
        # todo hardcode batch size = 1
        x_out = flash_attn_func(query[None, ...], key[None, ...], value[None, ...]) # add batch_size dim
        return x_out


class FlashAttnVarlen(BaseAttnVarlen):
    # flash attn params
    max_seqlen_q: int = None
    cu_seqlens_q: torch.Tensor = None
    max_seqlen_kv: int = None
    cu_seqlens_kv: torch.Tensor = None

    def self_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_idx: int = 0,
    ):
        x_out = flash_attn_varlen_func(
            query, key, value,
            cu_seqlens_q=FlashAttnVarlen.cu_seqlens_q,
            cu_seqlens_k=FlashAttnVarlen.cu_seqlens_q,
            max_seqlen_q=FlashAttnVarlen.max_seqlen_q,
            max_seqlen_k=FlashAttnVarlen.max_seqlen_q,
        )
        x_out = x_out[None, ...] # add batch_size dim
        return x_out

    def cross_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_idx: int = 0,
    ):
        x_out = flash_attn_varlen_func(
            query, key, value,
            cu_seqlens_q=FlashAttnVarlen.cu_seqlens_q,
            cu_seqlens_k=FlashAttnVarlen.cu_seqlens_kv,
            max_seqlen_q=FlashAttnVarlen.max_seqlen_q,
            max_seqlen_k=FlashAttnVarlen.max_seqlen_kv,
        )
        x_out = x_out[None, ...] # add batch_size dim
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_attn_params(
        n_seq: int = None,
        q_seq_len_list: List = None,
        kv_seq_len_list: List = None,
        torch_device: str = "cuda",
    ) -> None:
        seqlen_q = torch.tensor(q_seq_len_list, dtype=torch.int32, device=torch_device)
        cu_seqlens_q = torch.zeros(n_seq + 1, dtype=torch.int32, device=torch_device)
        cu_seqlens_q[1:] = torch.cumsum(seqlen_q, dim=0)
        FlashAttnVarlen.max_seqlen_q = max(seqlen_q)
        FlashAttnVarlen.cu_seqlens_q = cu_seqlens_q

        seqlen_kv = torch.tensor(kv_seq_len_list, dtype=torch.int32, device=torch_device)
        cu_seqlens_kv = torch.zeros(n_seq + 1, dtype=torch.int32, device=torch_device)
        cu_seqlens_kv[1:] = torch.cumsum(seqlen_kv, dim=0)
        FlashAttnVarlen.max_seqlen_kv = max(seqlen_kv)
        FlashAttnVarlen.cu_seqlens_kv = cu_seqlens_kv


class SageAttnVarlen(BaseAttnVarlen):
    # sage attn params
    max_seqlen_q: int = None
    cu_seqlens_q: torch.Tensor = None
    max_seqlen_kv: int = None
    cu_seqlens_kv: torch.Tensor = None

    def self_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_idx: int = 0,
    ):
        x_out = sageattn_varlen(
            query, key, value,
            cu_seqlens_q=SageAttnVarlen.cu_seqlens_q,
            cu_seqlens_k=SageAttnVarlen.cu_seqlens_q,
            max_seqlen_q=SageAttnVarlen.max_seqlen_q,
            max_seqlen_k=SageAttnVarlen.max_seqlen_q,
        )
        x_out = x_out[None, ...] # add batch_size dim
        return x_out

    def cross_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_idx: int = 0,
    ):
        x_out = sageattn_varlen(
            query, key, value,
            cu_seqlens_q=SageAttnVarlen.cu_seqlens_q,
            cu_seqlens_k=SageAttnVarlen.cu_seqlens_kv,
            max_seqlen_q=SageAttnVarlen.max_seqlen_q,
            max_seqlen_k=SageAttnVarlen.max_seqlen_kv,
        )
        x_out = x_out[None, ...] # add batch_size dim
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_attn_params(
        n_seq: int = None,
        q_seq_len_list: List = None,
        kv_seq_len_list: List = None,
        torch_device: str = "cuda",
    ) -> None:
        seqlen_q = torch.tensor(q_seq_len_list, dtype=torch.int32, device=torch_device)
        cu_seqlens_q = torch.zeros(n_seq + 1, dtype=torch.int32, device=torch_device)
        cu_seqlens_q[1:] = torch.cumsum(seqlen_q, dim=0)
        SageAttnVarlen.max_seqlen_q = max(seqlen_q)
        SageAttnVarlen.cu_seqlens_q = cu_seqlens_q

        seqlen_kv = torch.tensor(kv_seq_len_list, dtype=torch.int32, device=torch_device)
        cu_seqlens_kv = torch.zeros(n_seq + 1, dtype=torch.int32, device=torch_device)
        cu_seqlens_kv[1:] = torch.cumsum(seqlen_kv, dim=0)
        SageAttnVarlen.max_seqlen_kv = max(seqlen_kv)
        SageAttnVarlen.cu_seqlens_kv = cu_seqlens_kv


def build_attn_op(
    attn_mode: str = "flexflashattn_varlen",
    attn_type: str = "self-attention",
):
    if attn_mode == "flexflashattn_varlen":
        return FlexFlashAttnVarlen(attn_type)
    elif attn_mode == "flexflashattn_varlen_simple":
        return FlexFlashAttnVarlen_Simple(attn_type)
    elif attn_mode == "flashattn_varlen":
        return FlashAttnVarlen(attn_type)
    elif attn_mode == "sageattn_varlen":
        return SageAttnVarlen(attn_type)
    elif attn_mode == "sdpa":
        return torch_F.scaled_dot_product_attention
    elif attn_mode == "flash_attn":
        return flash_attn_func
    elif attn_mode == "sage_attn":
        return sageattn
    else:
        raise ValueError("Unknown attn mode")


def build_attn_mask(
    attn_mode: str = "flexflashattn_varlen",
    n_seq: int = None,
    q_seq_len_list: List = None,
    kv_seq_len_list: List = None,
    selfattn_mask_method: str = "cusum",
    latent_num_frames_list: List = None,
    latent_len_per_frame_list: List = None,
    latent_num_frames_per_chunk_list: List = None,
    device: str = "cuda",
    num_blocks: int = None,
    sparse_tf_start: int = 15,
    sparse_tf_end: int = -1,
    sparse_tf_last_n_clean_chunks: int = 1,
):
    if attn_mode == "flexflashattn_varlen":
        FlexFlashAttnVarlen.init_attn_params(
            n_seq,
            q_seq_len_list,
            kv_seq_len_list,
            selfattn_mask_method,
            latent_num_frames_list,
            latent_len_per_frame_list,
            latent_num_frames_per_chunk_list,
            device,
            num_blocks=num_blocks,
            sparse_tf_start=sparse_tf_start,
            sparse_tf_end=sparse_tf_end,
            sparse_tf_last_n_clean_chunks=sparse_tf_last_n_clean_chunks,
        )
    elif attn_mode == "flexflashattn_varlen_simple":
        pass
    elif attn_mode == "flashattn_varlen":
        FlashAttnVarlen.init_attn_params(n_seq, q_seq_len_list, kv_seq_len_list, device)
    elif attn_mode == "sageattn_varlen":
        SageAttnVarlen.init_attn_params(n_seq, q_seq_len_list, kv_seq_len_list, device)
    else:
        raise ValueError("Unknown attn mode")


if __name__ == '__main__':
    batch_size = 4
    q_seq_len = 4096
    kv_seq_len = 512
    num_head = 12
    head_dim = 128

    torch_device = 'cuda'
    torch_dtype = torch.bfloat16

    q = torch.randn(batch_size, num_head, q_seq_len, head_dim, dtype=torch_dtype, device=torch_device)
    k = torch.randn(batch_size, num_head, kv_seq_len, head_dim, dtype=torch_dtype, device=torch_device)
    v = torch.randn(batch_size, num_head, kv_seq_len, head_dim, dtype=torch_dtype, device=torch_device)

    # * flash attn batch
    flash_attn = build_attn_op(attn_mode="flash_attn")
    q_batch = rearrange(q, "b n s d -> b s n d")
    k_batch = rearrange(k, "b n s d -> b s n d")
    v_batch = rearrange(v, "b n s d -> b s n d")
    x_selfattn_batch = flash_attn(q_batch, q_batch, q_batch)
    x_crossattn_batch = flash_attn(q_batch, k_batch, v_batch)
    
    # * flash attn varlen
    attn_mode = "flexflashattn_varlen"
    q_seq_len_list = [q_seq_len] * batch_size
    kv_seq_len_list = [kv_seq_len] * batch_size
    build_attn_mask(
        attn_mode=attn_mode,
        n_seq=batch_size,
        q_seq_len_list=q_seq_len_list,
        kv_seq_len_list=kv_seq_len_list,
    )
    selfattn_varlen = build_attn_op(attn_mode=attn_mode, attn_type="self-attention")
    crossattn_varlen = build_attn_op(attn_mode=attn_mode, attn_type="cross-attention")
    
    q_varlen = rearrange(q, "b n s d -> (b s) n d")[None, ...]
    k_varlen = rearrange(k, "b n s d -> (b s) n d")[None, ...]
    v_varlen = rearrange(v, "b n s d -> (b s) n d")[None, ...]
    x_varlen_selfattn = selfattn_varlen(q_varlen, q_varlen, q_varlen)
    x_varlen_crossattn = crossattn_varlen(q_varlen, k_varlen, v_varlen)

    x_varlen_selfattn_gt = rearrange(x_selfattn_batch, "b s n d -> (b s) n d", b=batch_size, n=num_head)
    x_varlen_crossattn_gt = rearrange(x_crossattn_batch, "b s n d -> (b s) n d", b=batch_size, n=num_head)
    
    # * compute loss
    mean_diff_selfattn = (x_varlen_selfattn_gt - x_varlen_selfattn).abs().mean()
    mean_diff_crossattn = (x_varlen_crossattn_gt - x_varlen_crossattn).abs().mean()
    print(f"mean Abs Diff (flash self attention): {mean_diff_selfattn:.6f}")
    print(f"mean Abs Diff (flash cross attention): {mean_diff_crossattn:.6f}")
