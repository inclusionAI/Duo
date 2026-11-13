import torch
import imageio
import numpy as np

from typing import List, Union
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
import matplotlib.pyplot as plt
from einops import rearrange

from diffusers import (
    AutoencoderKLWan, 
    UniPCMultistepScheduler,
)
from diffusers.video_processor import VideoProcessor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)


# Import for FSDP state reset and mismatch detection
try:
    from torch.distributed.fsdp._fully_shard._fsdp_state import (
        TrainingState, 
        _get_module_fsdp_state
    )
    from torch.distributed.tensor import DTensor
    FSDP_STATE_AVAILABLE = True
except ImportError:
    FSDP_STATE_AVAILABLE = False
    DTensor = None


def _check_and_fix_fsdp_mismatch_for_block(module: torch.nn.Module):
    """
    Check and fix FSDP state/parameter mismatch for a transformer block
    and ALL its submodules that have their own FSDP state.
    
    This is critical for chunk-by-chunk training with activation checkpointing
    when FSDP is applied hierarchically (e.g., to attn1, attn2, ffn, and block).
    
    The problem occurs when:
    - Chunk N's backward completes, FSDP reshards parameters to DTensor
    - But training_state may still be PRE_BACKWARD
    - Chunk N-1's AC recompute starts, FSDP's _pre_forward sees PRE_BACKWARD
      and skips all-gather (assuming params are already unsharded)
    - But params are actually sharded DTensor, causing mixed Tensor/DTensor error
    
    Solution: Recursively detect when any submodule is in PRE_BACKWARD state 
    but has sharded params, then manually trigger unshard for that param group.
    """
    if not FSDP_STATE_AVAILABLE or DTensor is None:
        return
    
    # Recursively check and fix ALL submodules with FSDP state
    for submodule in module.modules():
        state = _get_module_fsdp_state(submodule)
        if state is None:
            continue
        
        # Only check when in PRE_BACKWARD state
        if state._training_state != TrainingState.PRE_BACKWARD:
            continue
        
        # Check if this module's FSDP param group has sharded params
        fsdp_param_group = state._fsdp_param_group
        if fsdp_param_group is None:
            continue
        
        # Check if any of this param group's managed params are still DTensor
        has_sharded_param = False
        for fsdp_param in fsdp_param_group.fsdp_params:
            if isinstance(fsdp_param.sharded_param, DTensor):
                has_sharded_param = True
                break
        
        if not has_sharded_param:
            continue
        
        # Mismatch detected! Need to manually unshard this param group
        # Temporarily set to IDLE to allow unshard
        fsdp_param_group._training_state = TrainingState.IDLE
        # Trigger unshard
        fsdp_param_group.unshard()
        fsdp_param_group.wait_for_unshard()
        # Set back to PRE_BACKWARD so post_forward will skip reshard
        fsdp_param_group._training_state = TrainingState.PRE_BACKWARD


def load_vae(
    vae_path: str,
    vae_type: str = "vdm_wan",
    torch_dtype: torch.dtype = None,
    torch_device: torch.device = None,
):
    if "vdm_wan" in vae_type:
        vae = AutoencoderKLWan.from_pretrained(
            vae_path,
            torch_dtype=torch_dtype,
            device_map="cpu"
        )
    else:
        raise ValueError("Unknown vae type")
    return vae.to(torch_device)


def load_text_encoder(
    text_encoder_path: str,
    text_encoder_type: str = "u5",
    torch_dtype: torch.dtype = None,
    torch_device: torch.device = None,
):
    if text_encoder_type == "u5":
        text_encoder = UMT5EncoderModel.from_pretrained(
            text_encoder_path,
            dtype=torch_dtype,
            device_map="cpu"
        )
    else:
        raise ValueError("Unknown text encoder type")
    return text_encoder.to(torch_device)


def load_scheduler(
    flow_shift: float = 5.0,
    scheduler_type: str = "multi_step",
):
    if scheduler_type == "multi_step":
        scheduler = UniPCMultistepScheduler(
            num_train_timesteps=1000,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=flow_shift,
        )
    else:
        raise ValueError("Unknown scheduler type")
    return scheduler


def load_tokenizer(
    tokenizer_path: str,
):
    tokenizer = T5TokenizerFast.from_pretrained(
        tokenizer_path,
    )
    return tokenizer


def convert_flow_pred_to_x0(flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor, scheduler) -> torch.Tensor:
    """
    Convert flow matching's prediction to x0 prediction.
    flow_pred: the prediction with shape [B, C, F, H, W]
    xt: the input noisy data with shape [B, C, F, H, W]
    timestep: scalar timestep

    pred = noise - x0
    x_t = (1-sigma_t) * x0 + sigma_t * noise
    we have x0 = x_t - sigma_t * pred
    """
    original_dtype = flow_pred.dtype
    flow_pred, xt, sigmas, timesteps = map(
        lambda x: x.double().to(flow_pred.device), [flow_pred, xt, scheduler.sigmas, scheduler.timesteps]
    )
    timestep_id = torch.argmin((timesteps - timestep).abs())
    sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
    x0_pred = xt - sigma_t * flow_pred
    return x0_pred.to(original_dtype)


def data_seq_to_patch(
    patch_size: list,
    data_seq: torch.Tensor,
    latent_num_frames: int,
    latent_height: int,
    latent_width: int,
    batch_size: int = 1,
):
    p_t, p_h, p_w = patch_size
    post_patch_num_frames = latent_num_frames // p_t
    post_patch_height = latent_height // p_h
    post_patch_width = latent_width // p_w

    data_patch = data_seq.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    data_patch = data_patch.permute(0, 7, 1, 4, 2, 5, 3, 6)
    data_patch = data_patch.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return data_patch

@torch.no_grad()
def prepare_latents(
    vae,
    text_encoder,
    videos: torch.Tensor,
    token_ids: torch.Tensor,
    token_mask: torch.Tensor = None,
    torch_dtype: torch.dtype = None,
    torch_device: torch.device = None,
):
    # note 1. prepare video latents
    latents = vae._encode(videos.to(device=torch_device, dtype=torch_dtype))
    mu, logvar = torch.chunk(latents, 2, dim=1)
    
    latents_mean = torch.tensor(vae.config.latents_mean).to(mu.device)
    latents_std = torch.tensor(vae.config.latents_std).to(mu.device)
    mu = normalize_latents(mu, latents_mean, 1.0 / latents_std)
    logvar = normalize_latents(logvar, latents_mean, 1.0 / latents_std)
    latents = torch.cat([mu, logvar], dim=1)

    posterior = DiagonalGaussianDistribution(latents)
    latents = posterior.sample()

    # note 2. prepare text latents
    input_ids = token_ids.squeeze(1)
    attention_mask = token_mask.squeeze(1) if token_mask is not None else None
    text_embs = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    return latents, text_embs


def normalize_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents = ((latents.float() - latents_mean) * latents_std).to(latents)
    return latents


def flow_match_xt(x0: torch.Tensor, n: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    r"""Forward process of flow matching."""
    return (1.0 - t) * x0 + t * n


def flow_match_target(n: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    r"""Loss target for flow matching."""
    return n - x0


def expand_tensor_dims(tensor: torch.Tensor, ndim: int) -> torch.Tensor:
    assert len(tensor.shape) <= ndim
    return tensor.reshape(tensor.shape + (1,) * (ndim - len(tensor.shape)))


def set_requires_grad(models: Union[torch.nn.Module, List[torch.nn.Module]], value: bool) -> None:
    if isinstance(models, torch.nn.Module):
        models = [models]
    for model in models:
        if model is not None:
            model.requires_grad_(value)


def save_video(
    video_frames: np.ndarray,
    output_path: str,
):
    video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]
    imageio.mimwrite(output_path, video_frames, fps=15)


@torch.no_grad()
def decode_latents_to_video(
    vae: AutoencoderKLWan,
    latents: torch.Tensor,
):
    if latents.ndim == 3:
        latents = latents[None, ...]
    latents = latents.to(vae.dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        1.0 / torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = latents / latents_std + latents_mean
    video = vae.decode(latents, return_dict=False)[0]
    video_processor = VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial)
    video = video_processor.postprocess_video(video, output_type="np")
    return video


def sample_timestep_id(
    batch_size: int = 1,
    weighting_scheme: str = "logit_normal",
    min_timestep_bd: float = 0.0,
    max_timestep_bd: float = 1.0,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    do_fix_timestep: bool = False,
    fix_timestep_weight: float = 0.2,
    num_train_timesteps: int = 1000,
):
    max_timestep_boundary = int(min(max_timestep_bd, 1.0) * num_train_timesteps)
    min_timestep_boundary = int(max(min_timestep_bd, 0.0) * num_train_timesteps)
    if weighting_scheme == "logit_normal":
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, [batch_size,]).long()
    else:
        raise RuntimeError("Unknown sample scheme!")

    return timestep_id


def adjust_latent_num_frames(
    latent_num_frames_lst: List[int],
    time_division_factor: int = 4,
    time_division_remainder: int = 1,
    latent_num_frames_per_chunk: int = -1,
):
    latent_num_frames_lst_adjusted = []
    for i in range(len(latent_num_frames_lst)):
        num_frames = int(latent_num_frames_lst[i])
        while (
            num_frames % time_division_factor != time_division_remainder or
            (latent_num_frames_per_chunk > 0 and num_frames % latent_num_frames_per_chunk != 0)
        ):
            num_frames -= 1
        latent_num_frames_lst_adjusted.append(num_frames)
    return latent_num_frames_lst_adjusted


def interpolate_camera_poses(
    src_indices: np.ndarray, 
    src_rot_mat: np.ndarray, 
    src_trans_vec: np.ndarray, 
    tgt_indices: np.ndarray,
) -> torch.Tensor:
        # interpolate translation
        interp_func_trans = interp1d(
            src_indices, 
            src_trans_vec, 
            axis=0, 
            kind='linear', 
            bounds_error=False,
            fill_value="extrapolate",
        )
        interpolated_trans_vec = interp_func_trans(tgt_indices)

        # interpolate rotaiton
        src_quat_vec = Rotation.from_matrix(src_rot_mat)
        # ensure there is no sudden change in qw
        quats = src_quat_vec.as_quat().copy()  # [N, 4]
        for i in range(1, len(quats)):
            if np.dot(quats[i], quats[i-1]) < 0:
                quats[i] = -quats[i]
        src_quat_vec = Rotation.from_quat(quats)
        slerp_func_rot = Slerp(src_indices, src_quat_vec)
        interpolated_rot_quat = slerp_func_rot(tgt_indices)
        interpolated_rot_mat = interpolated_rot_quat.as_matrix()

        poses = np.zeros((len(tgt_indices), 4, 4))
        poses[:, :3, :3] = interpolated_rot_mat
        poses[:, :3, 3] = interpolated_trans_vec
        poses[:, 3, 3] = 1.0
        return torch.from_numpy(poses).float()


def SE3_inverse(T: torch.Tensor) -> torch.Tensor:
    Rot = T[:, :3, :3] # [B,3,3]
    trans = T[:, :3, 3:] # [B,3,1]
    R_inv = Rot.transpose(-1, -2)
    t_inv = -torch.bmm(R_inv, trans)
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype)[None, :, :].repeat(T.shape[0], 1, 1)
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3:] = t_inv
    return T_inv


def compute_relative_poses(
    c2ws_mat: torch.Tensor, 
    framewise: bool = False, 
    normalize_trans: bool = True, 
) -> torch.Tensor:
    ref_w2cs = SE3_inverse(c2ws_mat[0:1])
    relative_poses = torch.matmul(ref_w2cs, c2ws_mat)
    # ensure identity matrix for 1st frame
    relative_poses[0] = torch.eye(4, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    if framewise:
        # compute pose between i and i+1
        relative_poses_framewise = torch.bmm(SE3_inverse(relative_poses[:-1]), relative_poses[1:])
        relative_poses[1:] = relative_poses_framewise
    if normalize_trans: # note refer to camctrl2: "we scale the coordinate inputs to roughly 1 standard deviation to simplify model learning."
        translations = relative_poses[:, :3, 3] # [f, 3]
        max_norm = torch.norm(translations, dim=-1).max()
        # only normlaize when moving
        if max_norm > 0:
            relative_poses[:, :3, 3] = translations / max_norm
    return relative_poses


@torch.no_grad()
def create_meshgrid(n_frames: int, height: int, width: int, bias: float = 0.5, device='cuda', dtype=torch.float32) -> torch.Tensor:
    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view([-1, 2]) + bias # [h*w, 2]
    grid_xy = grid_xy[None, ...].repeat(n_frames, 1, 1) # [f, h*w, 2]
    return grid_xy


def get_Ks_tramsformed(
    Ks: torch.Tensor,
    height_org: int,
    width_org: int,
    height_resize: int,
    width_resize: int,
    height_final: int,
    width_final: int,
):
    fx, fy, cx, cy = Ks.chunk(4, dim=-1) # [f, 1]

    scale_x = width_resize / width_org
    scale_y = height_resize / height_org

    fx_resize = fx * scale_x
    fy_resize = fy * scale_y
    cx_resize = cx * scale_x
    cy_resize = cy * scale_y

    crop_offset_x = (width_resize - width_final) / 2
    crop_offset_y = (height_resize - height_final) / 2

    cx_final = cx_resize - crop_offset_x
    cy_final = cy_resize - crop_offset_y
    
    Ks_transformed = torch.zeros_like(Ks)
    Ks_transformed[:, 0:1] = fx_resize
    Ks_transformed[:, 1:2] = fy_resize
    Ks_transformed[:, 2:3] = cx_final
    Ks_transformed[:, 3:4] = cy_final

    return Ks_transformed


def get_plucker_embeddings(
    c2ws_mat: torch.Tensor,
    Ks: torch.Tensor,
    height: int,
    width: int,
):
    n_frames = c2ws_mat.shape[0]
    device = c2ws_mat.device
    dtype = c2ws_mat.dtype

    # Grid: compute ONCE [h*w] instead of [f, h*w, 2] (saves ~240x memory)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing='ij',
    )
    x_flat = grid_x.reshape(-1)  # [h*w]
    y_flat = grid_y.reshape(-1)  # [h*w]

    # Directions: compute ONCE [h*w, 3] (intrinsics same for all frames, saves ~241x memory)
    fx, fy, cx, cy = Ks[0, 0], Ks[0, 1], Ks[0, 2], Ks[0, 3]
    dirs = torch.stack([
        (x_flat - cx) / fx,
        (y_flat - cy) / fy,
        torch.ones_like(x_flat),
    ], dim=-1)  # [h*w, 3]
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)  # [h*w, 3]

    # Rotate per frame: [f, 3, 3] @ [3, h*w] -> [f, 3, h*w] -> [f, h*w, 3]
    rays_d = (c2ws_mat[:, :3, :3] @ dirs.T).transpose(1, 2)  # [f, h*w, 3]
    # rays_o: use expand (lazy broadcast, no memory allocation)
    rays_o = c2ws_mat[:, :3, 3].unsqueeze(1).expand_as(rays_d)  # [f, h*w, 3]

    plucker_embeddings = torch.cat([rays_o, rays_d], dim=-1)  # [f, h*w, 6]
    return plucker_embeddings.view(n_frames, height, width, 6)


def get_camera_frustum(c2w, frustum_len=0.5, scale=1.0):
    W, H = scale, scale * 0.75
    Z = frustum_len
    points_cam = np.array([
        [0, 0, 0],
        [W, H, Z],
        [-W, H, Z],
        [-W, -H, Z],
        [W, -H, Z],
    ]).T
    R = c2w[:3, :3]
    T = c2w[:3, 3:4]
    points_world = (R @ points_cam) + T
    lines_indices = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]
    return points_world, lines_indices


def is_zero_wasd(
    wasd_action: np.ndarray,
):
    wasd_all_zero = False
    if wasd_action is not None:
        wasd_tensor = torch.from_numpy(wasd_action) if isinstance(wasd_action, np.ndarray) else wasd_action
        wasd_all_zero = bool((wasd_tensor == 0).all().item())
    return wasd_all_zero


def is_zero_rotation(
    c2ws: torch.Tensor,
    rot_threshold: float = 1e-3,
) -> bool:
    rot_mat = c2ws[:, :3, :3]

    # trace(R) = 1 + 2*cos(theta); identity -> trace=3, theta=0
    trace = rot_mat[:, 0, 0] + rot_mat[:, 1, 1] + rot_mat[:, 2, 2]  # [F-1]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    angles = torch.acos(cos_theta)  # [F-1], radians

    return bool((angles < rot_threshold).all().item())


def initialize_selfattn_kv_cache_lst(
    num_layers,
    cache_shape,
    dtype,
    device,
    sparse_start: int = -1,
    sparse_end: int = -1,
    deep_cache_shape: list = None,
):
    end = num_layers if sparse_end < 0 else sparse_end
    selfattn_kv_cache_lst = []
    for layer_idx in range(num_layers):
        if sparse_start >= 0 and sparse_start <= layer_idx < end:
            shape = deep_cache_shape
        else:
            shape = cache_shape
        selfattn_kv_cache_lst.append(
            {
                'k_buffer': torch.zeros(shape, dtype=dtype, device=device),
                'v_buffer': torch.zeros(shape, dtype=dtype, device=device),
                'start_idx': torch.tensor([0], dtype=torch.long, device=device),
            }
        )
    return selfattn_kv_cache_lst


def visualize_camera_trajectory(c2ws_tensor, output_file='camera_viz.mp4', fps=30):
    if isinstance(c2ws_tensor, torch.Tensor):
        c2ws = c2ws_tensor.detach().cpu().numpy()
    else:
        c2ws = c2ws_tensor

    num_frames = len(c2ws)
    positions = c2ws[:, :3, 3]

    cmap = plt.get_cmap("turbo")
    colors = [cmap(i / (num_frames - 1)) for i in range(num_frames)]

    fig = plt.figure(figsize=(9,9))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("white")
    ax.set_axis_off()

    min_bound = positions.min(axis=0)
    max_bound = positions.max(axis=0)
    center = (max_bound + min_bound) / 2
    range_span = (max_bound - min_bound).max() / 2
    ax.set_xlim(center[0]-range_span, center[0]+range_span)
    ax.set_ylim(center[1]-range_span, center[1]+range_span)
    ax.set_zlim(center[2]-range_span, center[2]+range_span)

    traj_segs = [ax.plot([],[],[], linewidth=2)[0] for _ in range(num_frames-1)]
    frustum_lines = [ax.plot([],[],[], linewidth=3.5)[0] for _ in range(8)]

    frames = []

    for frame_idx in range(num_frames):
        for i in range(frame_idx):
            seg = traj_segs[i]
            seg.set_data([positions[i,0], positions[i+1,0]],
                         [positions[i,1], positions[i+1,1]])
            seg.set_3d_properties([positions[i,2], positions[i+1,2]])
            seg.set_color(colors[i])

        c2w = c2ws[frame_idx]
        f_len = range_span * 0.18
        pts, indices = get_camera_frustum(c2w, frustum_len=f_len, scale=f_len*0.6)
        for line_obj, (i,j) in zip(frustum_lines, indices):
            line_obj.set_data([pts[0,i], pts[0,j]], [pts[1,i], pts[1,j]])
            line_obj.set_3d_properties([pts[2,i], pts[2,j]])
            line_obj.set_color(colors[frame_idx])

        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        image = image[:, :, :3]
        frames.append(image)

    imageio.mimsave(output_file, frames, fps=fps)
    plt.close()