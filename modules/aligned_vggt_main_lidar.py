import torch
from .load_fn import load_and_preprocess_images_from_frames
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import numpy as np
from .utils import *
class AlignedVGGTLidar:

    def __init__(self,model,block_size=10,keyframe_per_block=10,keyframe_memory_cap=10,depth_conf_threshold=1.0,near_plane=0.1,far_plane=5.0,device="cuda"):
        self.reference_extrinsic = None
        self.reference_idx = 0
        self.depth_conf_threshold = depth_conf_threshold
        self.device = device
        self.camera_poses = []
        self.world_camera_poses = []
        self.generated_pcd = []
        self.block_size = block_size # batch size
        self.keyframe_per_block = keyframe_per_block
        self.keyframe_memory_cap = keyframe_memory_cap #Sets when the system starts throwing away keyframes (FIFO) 
        self.keyframe_list = [] # Current list of keyframes
        self.keyframe_list_idx = [] # And their indices
        self.keyframe_indices = []
        self.generated_pcds = []
        self.camera_intrinsics = []
        self.frames = []
        self.images = []
        self.depth_masks = []
        self.depth_maps = []
        self.img_timestamps = []
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.model = model
        self.aggregated_tokens_list = None
        self.ps_idx = None
        self.processed_frames = 0
        self.near_plane = near_plane
        self.far_plane = far_plane

    def mean_rotation(self,Rs):
        M = Rs.sum(dim=0)                         # [3,3] simple sum of rotations
        U, _, Vh = torch.linalg.svd(M)                # M = U @ diag(S) @ Vh
        R = U @ Vh                                    # closest orthonormal matrix to M (projection)
        if torch.det(R) < 0:                          # if it's a reflection (det = -1)
            U[:, -1] *= -1                            # flip last column of U
            R = U @ Vh                                # recompute -> now det(R) should be +1
        return R                                      # valid rotation matrix in SO(3)
    def process_frames(self,frames, block_idx):
        with torch.inference_mode():
            # Select keyframes based on ORB
            frames = self.keyframe_list + frames
            self.processed_frames = len(frames)
            # Read input data
            images, lidar_depths, timestamps = load_and_preprocess_images_from_frames(frames, device = self.device)
            # Run dense predictions
            with torch.cuda.amp.autocast(dtype=self.dtype):
                images = images[None]
                self.aggregated_tokens_list, self.ps_idx = self.model.aggregator(images)
            pose_enc = self.model.camera_head(self.aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            vggt_depth_map, vggt_depth_conf = self.model.depth_head(self.aggregated_tokens_list, images, self.ps_idx)
            # Correct with known intrinsic
            extrinsic = extrinsic[0]
            frame_scales = []
            global_scale = 1.0
            # Filter out highly unstable points from the PCD, this usually removes shiny surfaces like windows 
            p = 10.0
            depth_mask = (vggt_depth_conf > self.depth_conf_threshold).squeeze(0)
            for i, idx in enumerate(self.keyframe_indices):
                depth_mask[i] = torch.tensor(self.depth_masks[idx],device=self.device)
            for i in range(len(self.keyframe_indices),vggt_depth_conf.shape[1]):
                conf = vggt_depth_conf[0, i].float()
                mask = conf > self.depth_conf_threshold
                if mask.sum() == 0:
                    mask = torch.ones_like(depth_mask[i])
                vals = conf[mask]
                lo = torch.quantile(vals, p / 100.0)        # pth percentile value
                percentile_mask = conf >= lo               # keep everything >= cutoff
                final_mask = mask & percentile_mask
                depth_mask[i] = final_mask
            combined_masks = []
            # For all frames in the block + keyframes
            frames_to_align = range(vggt_depth_map.shape[1])#
            if len(self.keyframe_indices) == 0:
                frames_to_align = range(vggt_depth_map.shape[1])
            for i in frames_to_align:
                lidar_depth = lidar_depths[i].squeeze()
                vggt_depth = vggt_depth_map[0][i].squeeze()
                lidar_mask = (lidar_depth > self.near_plane) & (lidar_depth < self.far_plane) & depth_mask[i]
                if lidar_mask.sum() == 0:
                    continue
                else:
                    # v2.0: least squares method based solution:
                    depth_meter = lidar_depth[lidar_mask]
                    depth_arbitrary = vggt_depth[lidar_mask]
                    # Solve min ||A*s - depth_meter||^2
                    num = (depth_arbitrary * depth_meter).sum()
                    den = (depth_arbitrary * depth_arbitrary).sum()

                    if den!=0:
                        frame_scales.append(num/den)
                    else:
                        print(f"WARNING: Depth for frame {i} dropped (zero denominator)")
            for i in range(vggt_depth_map.shape[1]):
                lidar_depth = lidar_depths[i].squeeze()
                lidar_mask = (lidar_depth > self.near_plane) & (lidar_depth < self.far_plane) & depth_mask[i]
                combined_masks.append(lidar_mask)
            combined_masks = torch.stack(combined_masks, dim=0)
            combined_masks = combined_masks.detach().cpu().numpy()
            if len(frame_scales) > 0:
                scales = torch.stack(frame_scales).detach().cpu().numpy()
                global_scale = float(np.median(scales))  #!!! torch.median != np.median
                vggt_depth_map[..., 0].mul_(global_scale)
            # We have the properly scaled depth maps here
            T_extrinsic = convert_to_4x4(extrinsic)
            # Use the scaling factor between the generated and LIDAR depth maps to rescale the extrinsic transformations
            # for all extrinsics in the block
            T_extrinsic[:, :3, 3].mul_(global_scale)
            if self.reference_extrinsic is not None:
                T_refs = convert_to_4x4(self.reference_extrinsics)          # [M,4,4]
                T_curr = T_extrinsic[self.reference_idxs]                   # [M,4,4]
                Xs = invert_extrinsic(T_curr) @ T_refs                      # [M,4,4], each is a candidate transform
                R = self.mean_rotation(Xs[:, :3, :3])
                t = Xs[:, :3, 3].median(dim=0).values                       # robust (matches your median style); use .mean(0) if preferred
                transform_matrix = torch.eye(4, device=T_extrinsic.device, dtype=T_extrinsic.dtype)
                transform_matrix[:3, :3] = R
                transform_matrix[:3, 3] = t
                T_extrinsic = T_extrinsic @ transform_matrix
            extrinsic = T_extrinsic[:, :3, :]
            cam_to_world_extrinsic = invert_extrinsic(T_extrinsic)[:,:3,:]
            # Project point cloud using the modified extrinsics
            per_frame_point_clouds = unproject_depth_map_to_point_map(vggt_depth_map.squeeze(0), extrinsic, intrinsic.squeeze(0))
            # Convert world to cam pose to cam to world pose for visualization
            for i in range(len(self.keyframe_list_idx),len(frames)):
                self.camera_poses.append(extrinsic[i].detach().cpu().numpy())
                self.world_camera_poses.append(cam_to_world_extrinsic[i].detach().cpu().numpy())
                self.generated_pcds.append(per_frame_point_clouds[i])
                self.depth_masks.append(combined_masks[i])
                self.images.append(images[0][i].detach().cpu().numpy())
                self.frames.append(frames[i]) # this also has the timestamp!
                self.camera_intrinsics.append(intrinsic[0][i].detach().cpu().numpy())
                self.depth_maps.append(vggt_depth_map[0][i].squeeze().detach().cpu().numpy())
            images, per_frame_point_clouds, combined_masks = images[0][len(self.keyframe_list_idx):], per_frame_point_clouds[len(self.keyframe_list_idx):], combined_masks[len(self.keyframe_list_idx):]
            self.update_keyframes_fixed(block_idx)
            if self.update_memory(block_idx,frames[len(self.keyframe_list_idx):]):
                self.reallocate_anchor()
            return images, per_frame_point_clouds, combined_masks
    def update_memory(self,block_idx, frames):
        # Update keyframes
        reallocate_anchor_frame = False
        for keyframe_idx in self.keyframe_indices:
            self.keyframe_list.append(frames[keyframe_idx - block_idx])
            self.keyframe_list_idx.append(keyframe_idx)
            if len(self.keyframe_list_idx) > self.keyframe_memory_cap:
                reallocate_anchor_frame = True
                self.keyframe_list = self.keyframe_list[1:]
                self.keyframe_list_idx = self.keyframe_list_idx[1:]
        return reallocate_anchor_frame
    def update_keyframes_fixed(self,block_idx):
        self.keyframe_indices = [i + block_idx for i in range(self.keyframe_per_block)]
    def reallocate_anchor(self):
        camera_pose_idx = self.keyframe_list_idx[0]
        self.reference_extrinsic = torch.tensor(self.camera_poses[camera_pose_idx],device=self.device)
        sel = np.arange(len(self.keyframe_list_idx), dtype=int)
        poses = np.stack([self.camera_poses[self.keyframe_list_idx[j]] for j in sel]).astype(np.float32)
        self.reference_extrinsics = torch.from_numpy(poses).to(self.device)
        self.reference_idxs = torch.as_tensor(sel, device=self.device, dtype=torch.long)
    def clean_memory(self):
        del self.aggregated_tokens_list, self.ps_idx
        torch.cuda.empty_cache()