import torch
import open3d as o3d
import numpy as np
import cv2 as cv
from .visualizer import *
class Static_Instance:
    def __init__(self,unique_id):
        self.colour = (0,0,0)
        self.mask = []
        self.pcd = []
        self.bounding_box = []
        self.oriented_bounding_box = []
        self.exclusion_list = []
        self.type_idx = -1
        self.temporal_index = -1
        self.unique_id = unique_id
        self.tracker_id_2d = -1 # GT for eval
        self.median = None
        self.mean = None

class SemanticSegmentator:
    def __init__(self,yolo_model,names,type_list,vggt_img_size,conf_threshold=0.4):
        self.yolo_model = yolo_model
        self.names = names
        self.conf_threshold = conf_threshold
        self.type_list = type_list
        self.yolo_predictions = {}
        self.vggt_img_size = vggt_img_size
        self.static_instances_per_type = {}
        self.static_instances_tracker_ids_2d = {}
        for type_idx in self.type_list:
            self.static_instances_per_type[type_idx] = {}
            self.static_instances_tracker_ids_2d[type_idx] = {}
        self.static_instances_per_frame = {}

    def segment_frames(self,frames, aligned_vggt,block_idx):
        static_instances,static_instance_ids,self.yolo_predictions = self.generate_static_instances_from_frames(frames, aligned_vggt, block_idx, self.type_list, self.yolo_model, self.yolo_predictions, self.vggt_img_size)
        for i in range(len(frames)):
            global_frame_id = i + block_idx
            self.static_instances_per_frame[global_frame_id] = {}
            for type_idx in self.type_list:
                self.static_instances_per_frame[global_frame_id][type_idx] = []
                self.static_instances_per_type[type_idx][global_frame_id]= []
                for id in static_instance_ids[type_idx][i]:
                    self.static_instances_per_type[type_idx][global_frame_id].append(static_instances[type_idx][id])
                    self.static_instances_per_frame[global_frame_id][type_idx].append(static_instances[type_idx][id])
    def clean_point_cloud(self,pcd):
        points_clean = np.asarray(pcd.points)
        if points_clean.shape[0] == 0:
            return None, None
        mins = points_clean.min(axis=0)
        maxs = points_clean.max(axis=0)
        return pcd,(mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2])
    def convert_to_obb(self, bbox):
        mins = np.asarray(bbox[:3], dtype=float)
        maxs = np.asarray(bbox[3:], dtype=float)
        return o3d.geometry.OrientedBoundingBox(center=((mins + maxs) * 0.5).tolist(),R=np.eye(3),extent=(maxs - mins).tolist())
    def generate_static_instances_from_frames(self,frames,aligned_vggt,block_idx,type_list,yolo_model,yolo_predictions,img_size):
        with torch.inference_mode():
            static_instances = {type_idx: [] for type_idx in type_list}
            static_instance_ids = {type_idx: [] for type_idx in type_list}
            kernel = np.ones((3, 3), np.uint8)
            depth_masks = aligned_vggt.depth_masks
            generated_pcds = aligned_vggt.generated_pcds
            for local_frame_idx, frame in enumerate(frames):
                frame_id = block_idx + local_frame_idx
                img = cv.cvtColor(frame[0], cv.COLOR_BGR2RGB)
                result = yolo_model.track(source=img, tracker="botsort.yaml", verbose=False, persist=True, conf=self.conf_threshold, iou=0.2, half=True)[0]
                for type_idx in type_list:
                    static_instance_ids[type_idx].append([])
                if result.boxes is None or result.masks is None:
                    continue
                boxes = result.boxes.cpu().numpy()
                masks = result.masks.data
                for det_idx in range(len(boxes)):
                    box = boxes[det_idx]
                    type_idx = int(box.cls.item())
                    if type_idx not in static_instances:
                        continue
                    # Erode mask
                    mask = masks[det_idx].to(torch.uint8).cpu().numpy()
                    mask = cv.resize(mask, img_size, interpolation=cv.INTER_NEAREST)
                    mask = cv.erode(mask * 255, kernel, iterations=1) > 0
                    # Create a local instance from the mask
                    static_inst = Static_Instance((frame_id << 20) | det_idx)
                    static_inst.type_idx = type_idx
                    static_inst.mask = mask
                    static_inst.temporal_index = frame_id
                    static_inst.tracker_id_2d = int(box.id.item()) if box.id is not None else -1
                    point_mask = np.bitwise_and(depth_masks[frame_id], mask)

                    original_pcd = o3d.geometry.PointCloud()
                    original_pcd.points = o3d.utility.Vector3dVector(generated_pcds[frame_id][point_mask].reshape(-1, 3))
                    static_inst.pcd, static_inst.bounding_box = self.clean_point_cloud(original_pcd)
                    # Only append to static instances if there are points in the cloud
                    if static_inst.pcd is None:
                        continue
                    static_inst.oriented_bounding_box = self.convert_to_obb(static_inst.bounding_box)
                    pts = np.asarray(static_inst.pcd.points)
                    static_inst.median = np.median(pts, axis=0)
                    static_inst.mean = np.mean(pts, axis=0)
                    static_instances[type_idx].append(static_inst)
                    static_instance_ids[type_idx][local_frame_idx].append(len(static_instances[type_idx]) - 1)
            torch.cuda.empty_cache()
            return static_instances,static_instance_ids,yolo_predictions