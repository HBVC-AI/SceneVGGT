import torch
import numpy as np
import cv2 as cv


def convert_to_4x4(extrinsics):
    batch = extrinsics.shape[:-2]
    b_row = torch.zeros(*batch, 1, 4, dtype=extrinsics.dtype, device=extrinsics.device)
    b_row[..., 0, -1] = 1
    return torch.cat([extrinsics,b_row],dim=-2)
def invert_extrinsic(T):
    # We want
    # T^-1 = [R^T -R^T*t]
    #        [0    1]
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    R_transposed = R.transpose(-1, -2)
    ti = -(R_transposed @ t.unsqueeze(-1)).squeeze(-1)
    T_inv = torch.zeros_like(T)
    T_inv[..., :3, :3] = R_transposed
    T_inv[..., :3, 3]  = ti
    T_inv[..., 3, 3]   = 1.0
    return T_inv
def select_keyframes_orb(block_idx,frame_list,keyframe_per_block,orb, allow_zero = False):
    keypoint_cnt = []
    for img_path in frame_list:
        img = cv.imread(img_path[0]) # image_and_depth_names = [frame_path, depth_path]
        keypoints,_ = orb.detectAndCompute(img,None)
        keypoint_cnt.append(len(keypoints))
    if allow_zero:
        return np.argpartition(np.array(keypoint_cnt),-keyframe_per_block)[-keyframe_per_block:]+block_idx
    else:
        return np.argpartition(np.array(keypoint_cnt[1:]),-keyframe_per_block)[-keyframe_per_block:]+block_idx+1
def select_keyframes_orb_from_frames(block_idx,frames,keyframe_per_block,orb, allow_zero = False):
    keypoint_cnt = []
    for img in frames:
        keypoints,_ = orb.detectAndCompute(img[0],None)
        keypoint_cnt.append(len(keypoints))
    print(keypoint_cnt)
    if allow_zero:
        return np.argpartition(np.array(keypoint_cnt),-keyframe_per_block)[-keyframe_per_block:]+block_idx
    else:
        return np.argpartition(np.array(keypoint_cnt[1:]),-keyframe_per_block)[-keyframe_per_block:]+block_idx+1
def select_keyframes_orb_from_frames_uniform(block_idx, frames, keyframe_per_block, orb, allow_zero=False):
    cnt = [len(orb.detectAndCompute(f[0], None)[0]) for f in frames]

    idx = np.arange(0 if allow_zero else 1, len(frames))
    if keyframe_per_block <= 0 or idx.size == 0:
        return np.array([], dtype=int)
    if keyframe_per_block >= idx.size:
        return idx + block_idx

    picks = []
    for seg in np.array_split(idx, keyframe_per_block):
        picks.append(seg[np.argmax([cnt[i] for i in seg])])
    return np.array(picks, dtype=int) + block_idx
def select_keyframes_orb_from_frames_ind(block_idx,frames,keyframe_per_block,orb, allow_zero = False):
    keypoint_cnt = []
    kp_pixels = []
    for img in frames:
        keypoints,_ = orb.detectAndCompute(img[0],None)
        if keypoints is None:
            keypoints = []
        keypoint_cnt.append(len(keypoints))
        kp_pixels.append(np.rint([kp.pt for kp in keypoints]).astype(np.int32))
    if allow_zero:
        sel = np.argpartition(np.array(keypoint_cnt),-keyframe_per_block)[-keyframe_per_block:]+block_idx
        return sel, [kp_pixels[i - block_idx] for i in sel]
    else:
        sel = np.argpartition(np.array(keypoint_cnt[1:]),-keyframe_per_block)[-keyframe_per_block:]+block_idx+1
        return sel, [kp_pixels[i - block_idx] for i in sel]
def select_keyframes_orb_from_frames_ind(block_idx,frames,keyframe_per_block,orb, allow_zero = False, target_hw=None):
    keypoint_cnt = []
    kp_pixels = []
    for img in frames:
        keypoints,_ = orb.detectAndCompute(img[0],None)
        if keypoints is None:
            keypoints = []
        keypoint_cnt.append(len(keypoints))
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        if target_hw is not None and pts.size:
            Hs, Ws = img[0].shape[:2]
            Ht, Wt = target_hw
            pts[:, 0] *= (Wt / Ws)
            pts[:, 1] *= (Ht / Hs)
        kp_pixels.append(np.rint(pts).astype(np.int32))
    if allow_zero:
        sel = np.argpartition(np.array(keypoint_cnt),-keyframe_per_block)[-keyframe_per_block:]+block_idx
        return sel, [kp_pixels[i - block_idx] for i in sel]
    else:
        sel = np.argpartition(np.array(keypoint_cnt[1:]),-keyframe_per_block)[-keyframe_per_block:]+block_idx+1
        return sel, [kp_pixels[i - block_idx] for i in sel]

def select_keyframes_depth(block_idx, depth_maps, keyframe_per_block, allow_zero=False):
    def score(d):
        gx = np.diff(d, axis=1) # difference of neighbouring values along the axis ~ basically the gradient
        gy = np.diff(d, axis=0)
        return float(np.mean(gx[:-1]**2 + gy[:, :-1]**2))  # magnitudes

    scores = np.array([score(d) for d in depth_maps], dtype=np.float32)
    if allow_zero:
        return np.argpartition(scores,-keyframe_per_block)[-keyframe_per_block:]+block_idx
    else:
        return np.argpartition(scores[1:],-keyframe_per_block)[-keyframe_per_block:]+block_idx+1
def select_keyframes_confidence(block_idx, depth_confidences, keyframe_per_block, allow_zero=False):
    scores = torch.stack([d.sum() for d in depth_confidences]).float()
    if allow_zero:
        idx = torch.topk(scores, k=keyframe_per_block).indices
        return (idx + block_idx).cpu().numpy()
    else:
        idx = torch.topk(scores[1:], k=keyframe_per_block).indices + 1
        return (idx + block_idx).cpu().numpy()