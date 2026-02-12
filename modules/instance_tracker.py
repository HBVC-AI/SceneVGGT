import torch
import numpy as np
from .visualizer import *
from .semantic_segmentator import *
import time
import cv2
class InstanceTracker:
    def __init__(self,vggt_img_size,type_list):
        self.track_count = {}
        self.per_frame_instance_masks = {}
        self.per_frame_instance_tracker_id_2d = {}
        self.associated_global_masks = {}
        self.subtracklets = {}
        self.type_list = type_list
        self.img_size = vggt_img_size
        for type_idx in type_list:
            self.track_count[type_idx] = 0
            self.per_frame_instance_masks[type_idx] = []
            self.per_frame_instance_tracker_id_2d[type_idx] = []
            self.associated_global_masks[type_idx] = []
            self.subtracklets[type_idx] = {}
        rows, cols = np.meshgrid(np.arange(0,vggt_img_size[1],20), np.arange(0,vggt_img_size[0],20), indexing='ij')
        self.mgrid = np.stack((rows, cols), axis=-1).reshape(-1, 2)
    def associate_local_to_global(self,associated_local_masks,tracked_frame_indices,type_idx):
        # Which masks do the points belong to on the first frame, this will set the initial track 
        initial_track_loyalties = associated_local_masks[0]
        # How many unique masks/subtracklets are there
        initial_mask_count = int(np.amax(np.unique(initial_track_loyalties)))
        for i in tracked_frame_indices:
            if i >= len(self.associated_global_masks[type_idx]):
                self.associated_global_masks[type_idx].append([])
        conversion_dict = {}
        # Iterate over all frames
        for i in range(len(tracked_frame_indices)):
            frame_id = tracked_frame_indices[i]
            votes = np.zeros((self.per_frame_instance_masks[type_idx][frame_id].shape[0]+1,initial_mask_count+1))
            # Iterate over all valid tracked points on this frame
            for j in range(associated_local_masks.shape[1]):
                # If that points intersects the local mask K, place a vote linking mask K, with the initial track of that point
                if associated_local_masks[i,j] == 0:
                    continue
                votes[int(associated_local_masks[i,j]),int(initial_track_loyalties[j])] += 1
            per_row_max_votes = np.argmax(votes,axis=1) #Per row
            per_row_max_vote_values = np.amax(votes,axis=1)
            per_row_single_votes = np.zeros((self.per_frame_instance_masks[type_idx][frame_id].shape[0]+1,initial_mask_count+1))
            for f in range(len(per_row_max_votes)):
                per_row_single_votes[f][per_row_max_votes[f]] = per_row_max_vote_values[f]
            per_column_max_votes = np.argmax(per_row_single_votes,axis=0)
            per_column_max_vote_values = np.amax(per_row_single_votes,axis=0)
            final_votes = np.zeros_like(per_row_max_votes)
            final_vote_values = np.zeros_like(per_row_max_votes)
            for f in range(per_column_max_votes.shape[0]):
                if per_column_max_vote_values[f] != 0:
                    final_votes[per_column_max_votes[f]] = f
                    final_vote_values[per_column_max_votes[f]] = per_column_max_vote_values[f]
            if i == 0:
                temporary_global_masks = []
                for k in range(1,final_votes.shape[0]):
                    temporary_global_masks.append(final_votes[k])
                if len(self.associated_global_masks[type_idx][tracked_frame_indices[i]]) == 0:
                    for e in range(len(temporary_global_masks)):# Everything is unaccounted for
                        self.associated_global_masks[type_idx][tracked_frame_indices[i]].append(self.track_count[type_idx])
                        self.subtracklets[type_idx][self.track_count[type_idx]] = [(tracked_frame_indices[i],e+1,True)]
                        self.track_count[type_idx] += 1
                        conversion_dict[temporary_global_masks[e]] = self.associated_global_masks[type_idx][tracked_frame_indices[i]][e]
                else:
                    for e in range(len(temporary_global_masks)):
                        if self.associated_global_masks[type_idx][tracked_frame_indices[i]][e] == 0: #Unaccounted for mask
                            self.associated_global_masks[type_idx][tracked_frame_indices[i]][e] = self.track_count[type_idx]
                            self.subtracklets[type_idx][self.track_count[type_idx]] = [(tracked_frame_indices[i],e+1,True)]
                            self.track_count[type_idx] += 1
                        conversion_dict[temporary_global_masks[e]] = self.associated_global_masks[type_idx][tracked_frame_indices[i]][e]
                continue
            self.associated_global_masks[type_idx][tracked_frame_indices[i]] = []
            for k in range(1,final_votes.shape[0]):
                if final_votes[k] in conversion_dict:
                    self.associated_global_masks[type_idx][frame_id].append(conversion_dict[final_votes[k]])
                    self.subtracklets[type_idx][conversion_dict[final_votes[k]]].append((tracked_frame_indices[i],k,True))
                else:
                    self.associated_global_masks[type_idx][frame_id].append(final_votes[k])
                    if final_votes[k] not in self.subtracklets[type_idx]:
                        self.subtracklets[type_idx][final_votes[k]] = []
                    self.subtracklets[type_idx][final_votes[k]].append((tracked_frame_indices[i],k,True))
        if self.track_count[type_idx] == 0:
            self.track_count[type_idx] = np.amax(np.array(list(self.subtracklets[type_idx].keys()))) + 1
            
    def track_instances(self,aligned_vggt, tracked_frame_indices):
        with torch.inference_mode():
            tracked_images = torch.stack([torch.tensor(aligned_vggt.images[i], device=aligned_vggt.device) for i in tracked_frame_indices],dim=0)
            # Created tracked points based on each mask
            slices = {}
            pts_all = []
            off = 0
            
            for type_idx in self.type_list:
                masks_anchor_frame = self.per_frame_instance_masks[type_idx][tracked_frame_indices[0]]
                if len(masks_anchor_frame) == 0:
                    continue

                tracked_pts = []
                for m_idx in range(masks_anchor_frame.shape[0]):
                    pts = self.mgrid[masks_anchor_frame[m_idx][self.mgrid[:, 0], self.mgrid[:, 1]]].reshape(-1, 2)
                    if pts.size:
                        tracked_pts.append(pts)

                if not tracked_pts:
                    continue
                tracked_pts = np.concatenate(tracked_pts, axis=0)[:, ::-1]  # (x,y)
                slices[type_idx] = (off, off + tracked_pts.shape[0])
                off += tracked_pts.shape[0]
                pts_all.append(tracked_pts)
            if not pts_all:
                return
            tracked_points = np.concatenate(pts_all, axis=0)
            with torch.no_grad():
                query_points = torch.FloatTensor(tracked_points.copy()).to(aligned_vggt.device)
                track_list, vis_score, conf_score = aligned_vggt.model.track_head(aligned_vggt.aggregated_tokens_list,tracked_images.unsqueeze(0),aligned_vggt.ps_idx,query_points=query_points[None],query_index=0)
            track_list = track_list[-1][0].detach().cpu().numpy()
            conf_score = conf_score[0].detach().cpu().numpy()  

            for type_idx, (s_start, s_end) in slices.items():
                associated_local_masks = np.zeros((len(tracked_frame_indices), s_end - s_start))
                # For each frame and each point determine, which local mask they overlap with
                # For this each tracked point that has a high enough confidence and fallls within the frame, we intersect with the masks
                for i, frame_id in enumerate(tracked_frame_indices):
                    masks = self.per_frame_instance_masks[type_idx][frame_id]
                    if len(masks) == 0:
                        continue
                    for j in range(s_start, s_end):
                        x, y = track_list[i][j]
                        if (conf_score[i][j] > 0.1 and x >= 0 and x < self.img_size[0] and y >= 0 and y < self.img_size[1]):
                            if np.sum(masks[:, int(y), int(x)]) > 0:
                                associated_local_masks[i, j - s_start] = np.argmax(masks[:, int(y), int(x)]) + 1
                self.associate_local_to_global(associated_local_masks,tracked_frame_indices,type_idx)
            del track_list, tracked_images, vis_score, conf_score
            torch.cuda.empty_cache()
            
    def run_instance_tracking(self, aligned_vggt, semantic_segmentator, tracked_frame_indices):
        any_masks = False
        # Go over each type and frame
        for type_idx in self.type_list:
            for i in tracked_frame_indices:
                if i >= len(self.per_frame_instance_masks[type_idx]):
                    self.per_frame_instance_masks[type_idx].append([]) # Add an empty mask list for this frame
                    for k in range(len(semantic_segmentator.static_instances_per_frame[i][type_idx])): # Go over the masks associated to this type and frame and add them to the list
                        self.per_frame_instance_masks[type_idx][i].append(semantic_segmentator.static_instances_per_frame[i][type_idx][k].mask)
                    self.per_frame_instance_masks[type_idx][i] = np.asarray(self.per_frame_instance_masks[type_idx][i])

                if i >= len(self.associated_global_masks[type_idx]):
                    self.associated_global_masks[type_idx].append([])

            if len(self.per_frame_instance_masks[type_idx][tracked_frame_indices[0]]) > 0:
                any_masks = True
        if any_masks:
            self.track_instances(aligned_vggt, tracked_frame_indices)
