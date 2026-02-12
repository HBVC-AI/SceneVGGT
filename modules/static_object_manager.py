import open3d as o3d
import numpy as np
from modules.utils import *
import time


class Static_Object:
    def __init__(self):
        self.colour = (0,0,0)
        self.pcd = o3d.geometry.PointCloud()
        self.original_pcd = o3d.geometry.PointCloud()
        self.bounding_box = None
        self.oriented_bounding_box = None
        self.instance_indices = []
        self.type_idx = -1
        self.temporal_index = -1
        self.active = True
        self.status = 2 # 2: Fresh data|1: Memory|0: Discarded
        self.memory_confidence = 1.0

class StaticObjectManager:
    def __init__(self,type_list,vggt_img_size,memory_decay_rate = 0.1):
        self.type_list = type_list
        self.vggt_img_size = vggt_img_size
        self.object_tracks = {}
        self.objects = {}
        self.discarded_objects = {}
        self.discarded_object_tracks = {}
        self.fg_memory_xyz = np.asarray([],dtype=np.float32)
        self.fg_memory_rgb = np.asarray([],dtype=np.uint8)
        self.bg_memory_xyz = np.asarray([],dtype=np.float32)
        self.bg_memory_rgb = np.asarray([],dtype=np.uint8)
        self.memory_decay_rate = memory_decay_rate
        for type_idx in type_list:
            self.object_tracks[type_idx] = {}
            self.objects[type_idx] = {}
            self.discarded_object_tracks[type_idx] = {}
            self.discarded_objects[type_idx] = {}
    def convert_to_obb(self,bbox):
        extent = [(bbox[3] - bbox[0]),(bbox[4] - bbox[1]),(bbox[5] - bbox[2])]
        center = [(bbox[3] + bbox[0])/2.0,(bbox[4] + bbox[1])/2.0,(bbox[5] + bbox[2])/2.0]
        rotation = np.eye(3)
        return o3d.geometry.OrientedBoundingBox(center=center, R=rotation, extent=extent)
    def chamfer_distance(self,pcd1, pcd2):
        pcd1_tree = o3d.geometry.KDTreeFlann(pcd1)
        pcd2_tree = o3d.geometry.KDTreeFlann(pcd2)
        def avg_closest_dist(src, dst_tree):
            dists = []
            for pt in src.points:
                [_, idx, dist] = dst_tree.search_knn_vector_3d(pt, 1)
                dists.append(np.sqrt(dist[0]))
            return np.mean(dists)

        d1 = avg_closest_dist(pcd1, pcd2_tree)
        d2 = avg_closest_dist(pcd2, pcd1_tree)
        return np.amin([d1, d2])
    def calculate_bounding_boxes(self,obj):
        return self.calculate_bounding_boxes_pcd(obj.pcd)
    def calculate_bounding_boxes_pcd(self,pcd):
        segmented_cleaned = np.asarray(pcd.points)
        bbox = (np.amin(segmented_cleaned[:,0]),np.amin(segmented_cleaned[:,1]),np.amin(segmented_cleaned[:,2]),np.amax(segmented_cleaned[:,0]),np.amax(segmented_cleaned[:,1]),np.amax(segmented_cleaned[:,2]))
        obbox = self.convert_to_obb(bbox)
        return bbox,obbox
    def calculate_closeness_metric(self,type_idx,new_object_idx,old_object_indices,instance_tracker,semantic_segmentator):
        closeness_values = []
        new_medians = []
        for k in range(len(instance_tracker.subtracklets[type_idx][new_object_idx])):
            frame_id, mask_id,_ = instance_tracker.subtracklets[type_idx][new_object_idx][k]
            instance = semantic_segmentator.static_instances_per_frame[frame_id][type_idx][mask_id-1]
            new_medians.append(instance.mean)
        new_medians = np.asarray(new_medians)
        new_med_pcd = o3d.geometry.PointCloud()
        new_med_pcd.points = o3d.utility.Vector3dVector(new_medians)
        for i in old_object_indices:
            old_medians = []
            for k in range(len(instance_tracker.subtracklets[type_idx][i])):
                frame_id, mask_id,_ = instance_tracker.subtracklets[type_idx][i][k]
                instance = semantic_segmentator.static_instances_per_frame[frame_id][type_idx][mask_id-1]
                old_medians.append(instance.mean)
            old_medians = np.asarray(old_medians)
            old_med_pcd = o3d.geometry.PointCloud()
            old_med_pcd.points = o3d.utility.Vector3dVector(old_medians)
            chdist = self.chamfer_distance(new_med_pcd,old_med_pcd)
            closeness_values.append(chdist)
        return np.asarray(closeness_values)
    def should_be_visible(self,pcd, camera_extrinsic, camera_intrinsic, depth_map, img_size, infov_threshold=0.5):
        W, H = img_size
        points = np.asarray(pcd.points)
        num_points = points.shape[0]
        homogeneous_points = np.hstack([points, np.ones((num_points,1), dtype=points.dtype)])
        # World -> camera
        camera_points_h = (camera_extrinsic @ homogeneous_points.T).T
        camera_points = camera_points_h[:, :3]
        Z = camera_points[:, 2]
        # Keep only points in front of camera
        front = Z > 0
        # Project to pixels:
        image_points = (camera_intrinsic @ camera_points.T).T
        u = image_points[:, 0] / Z
        v = image_points[:, 1] / Z
        # Inside image bounds 
        in_bounds = (u >= 0) & (v >= 0) & (u < W) & (v < H)
        mask = front & in_bounds
        #Check for occlusion
        ui = np.clip(u[mask].astype(int), 0, W-1) # For numerical stability, avoid 518.00001 
        vi = np.clip(v[mask].astype(int), 0, H-1)
        eps = 1e-3 # Tolerance
        visible = Z[mask] <= depth_map[vi, ui] + eps
        full_visible = np.zeros_like(mask)
        full_visible[np.where(mask)[0]] = visible
        mask = full_visible
        ratio = mask.sum() / max(num_points, 1)
        return (ratio > infov_threshold)
    def refresh_temporality(self,aligned_vggt,instance_tracker,tracked_frame_indices):
        first_frame_idx = tracked_frame_indices[-aligned_vggt.block_size]
        for type_idx in self.type_list:
            objects_with_fresh_data = 0
            visible_objects = 0
            not_visible_objects = 0
            discarded_object_ids = []
            if self.memory_decay_rate > 0.001:
                for k,v in self.objects[type_idx].items():
                    if v.temporal_index >= first_frame_idx:
                        v.memory_confidence = 1.0
                        v.status = 2
                        objects_with_fresh_data += 1
                    else:
                        if self.should_be_visible(v.pcd,aligned_vggt.camera_poses[tracked_frame_indices[-1]],aligned_vggt.camera_intrinsics[tracked_frame_indices[-1]],aligned_vggt.depth_maps[tracked_frame_indices[-1]],self.vggt_img_size):
                            visible_objects += 1
                            v.status = 1
                            v.memory_confidence -= self.memory_decay_rate
                            if v.memory_confidence <= 0.00001:
                                v.status = 0
                                v.active = False
                                print(f"Discarding object: {k}")
                                discarded_object_ids.append(k)
                        else:
                            not_visible_objects += 1
                            v.status = 1
            for doi in discarded_object_ids:
                self.discarded_objects[type_idx][doi] = self.objects[type_idx][doi]
                self.discarded_object_tracks[type_idx][doi] = self.object_tracks[type_idx][doi]
                id_col = np.asarray([type_idx, doi // 255, doi % 255],dtype=np.uint8)
                mask = ~np.all(self.fg_memory_rgb == id_col, axis=-1)
                self.fg_memory_rgb = self.fg_memory_rgb[mask]
                self.fg_memory_xyz = self.fg_memory_xyz[mask]
                del self.objects[type_idx][doi]
                del self.object_tracks[type_idx][doi]
                for i in range(len(instance_tracker.subtracklets[type_idx][doi])):
                    instance_tracker.subtracklets[type_idx][doi][i] = (instance_tracker.subtracklets[type_idx][doi][i][0],instance_tracker.subtracklets[type_idx][doi][i][1],False)
    def update_objects(self,aligned_vggt,instance_tracker,semantic_segmentator,tracked_frame_indices):
        for type_idx in self.type_list:
            new_object_ids = []
            old_object_ids = []
            reidentifications = []
            for k, v in instance_tracker.subtracklets[type_idx].items():
                if k == 0:
                    continue
                if k not in self.object_tracks[type_idx]:
                    self.object_tracks[type_idx][k] = list(v)
                    self.objects[type_idx][k] = Static_Object()
                    new_object_ids.append(k)
                    for f in v:
                        frame_id, mask_id, _ = f
                        if frame_id > self.objects[type_idx][k].temporal_index:
                            self.objects[type_idx][k].temporal_index = frame_id
                        self.objects[type_idx][k].pcd += semantic_segmentator.static_instances_per_frame[frame_id][type_idx][mask_id-1].pcd
                else:
                    old_object_ids.append(k)
                    if len(v) != len(self.object_tracks[type_idx][k]):
                        for f in v[len(self.object_tracks[type_idx][k]):]:
                            frame_id, mask_id, _ = f
                            if frame_id > self.objects[type_idx][k].temporal_index:
                                self.objects[type_idx][k].temporal_index = frame_id
                            self.objects[type_idx][k].pcd += semantic_segmentator.static_instances_per_frame[frame_id][type_idx][mask_id-1].pcd
                        self.object_tracks[type_idx][k] += v[len(self.object_tracks[type_idx][k]):]
                self.objects[type_idx][k].bounding_box, self.objects[type_idx][k].oriented_bounding_box = self.calculate_bounding_boxes(self.objects[type_idx][k])
            if len(old_object_ids) > 0:
                for idx in new_object_ids:
                    closeness_values = self.calculate_closeness_metric(type_idx,idx,old_object_ids,instance_tracker,semantic_segmentator)
                    if np.amin(closeness_values) < 0.3:
                        reidentifications.append((old_object_ids[np.argmin(closeness_values)],idx))
            for reid in reidentifications:
                self.objects[type_idx][reid[0]].pcd += self.objects[type_idx][reid[1]].pcd
                self.object_tracks[type_idx][reid[0]] += self.object_tracks[type_idx][reid[1]]
                self.objects[type_idx][reid[0]].bounding_box, self.objects[type_idx][reid[0]].oriented_bounding_box = self.calculate_bounding_boxes(self.objects[type_idx][reid[0]])
                instance_tracker.subtracklets[type_idx][reid[0]] += instance_tracker.subtracklets[type_idx][reid[1]]
                for frame_id in range(len(instance_tracker.associated_global_masks[type_idx])):
                    for e in range(len(instance_tracker.associated_global_masks[type_idx][frame_id])):
                        if instance_tracker.associated_global_masks[type_idx][frame_id][e] == reid[1]:
                            if frame_id > self.objects[type_idx][reid[0]].temporal_index:
                                self.objects[type_idx][reid[0]].temporal_index = frame_id
                            instance_tracker.associated_global_masks[type_idx][frame_id][e] = reid[0]
                self.objects[type_idx][reid[0]].memory_confidence = max(self.objects[type_idx][reid[0]].memory_confidence,self.objects[type_idx][reid[1]].memory_confidence)
                print(f"Reidentification: {type_idx} : {reid[1]} -> {reid[0]}")
                id_col = np.asarray([type_idx, reid[1] // 255, reid[1] % 255],dtype=np.uint8)
                mask = np.all(self.fg_memory_rgb == id_col, axis=-1)
                self.fg_memory_rgb[mask] = (type_idx, reid[0] // 255, reid[0] % 255)
                del self.object_tracks[type_idx][reid[1]]
                del self.objects[type_idx][reid[1]]
                del instance_tracker.subtracklets[type_idx][reid[1]]
        self.refresh_temporality(aligned_vggt,instance_tracker,tracked_frame_indices)
        return instance_tracker