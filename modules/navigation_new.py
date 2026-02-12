import open3d as o3d
import numpy as np
from .utils import *
import copy
import json
import types
import math
import os
import time


class NavMapGenerator:
    def __init__(self,res,min_height=0.1,max_height=2.0,normal_min_height=0.2,normal_threshold=0.8):

        self.cast_time_sum = 0
        self.cast_time_n = 0

        self.res = res
        self.abs = None
        self.count = None
        self.min = None
        self.max = None
        self.semantic = None

        self.last_semantics = 0


        self.bbox = np.array([float('inf'),float('inf'),float('-inf'),float('-inf')])
        self.w = 0
        self.h = 0
        #self.points_all = np.empty((0,3))

        #these are used for filtering
        self.up = np.array([0,0,1]) #up vector, ground plane should match this, keep normalized
        self.min_height = min_height #discard point below this
        self.max_height = max_height #discard point above this
        self.normal_min_height = normal_min_height #discard points below this, if their normal matches the up vector
        self.normal_threshold = normal_threshold #if we take dot product of surface normal and up, and abs of result is above this, then those points match the plane normal
        #self.camera_pos = np.empty([0,3])
        #self.camera_forward = np.empty([0,3])

    #having the range be larger than needed is fine, so I'll snap it to grid
    #this is done to avoid mismatch when we aggregate maps
    def snap_lower(self,x):
            if x <= 0:
                x-=self.res/2
            else:
                x+=self.res/2
            return self.res*(math.trunc(x/self.res))-self.res/2
    
    def snap_upper(self,x):
        if x <= 0:
            x-=self.res/2
        else:
            x+=self.res/2
        return self.res*(math.trunc(x/self.res))+self.res/2

    def get_bbox(self,points):
        xmin = self.bbox[0]
        ymin = self.bbox[1]
        xmax = self.bbox[2]
        ymax = self.bbox[3]

        xmin = min(xmin,np.min(points[:,0]))
        ymin = min(ymin,np.min(points[:,1]))
        xmax = max(xmax,np.max(points[:,0]))
        ymax = max(ymax,np.max(points[:,1]))
        nbbox = np.array([self.snap_lower(xmin),self.snap_lower(ymin),self.snap_upper(xmax),self.snap_upper(ymax)])
        return nbbox
    
    def expand_map(self,old_map,new_bbox,nw,nh,def_val=0,dtype=np.float64):
        new_map = np.full((nh,nw),def_val,dtype=dtype)
        #print(self.bbox)
        #print(new_bbox)
        if old_map is not None:
            sx = int((self.bbox[0]-new_bbox[0])//self.res)
            sy = int((self.bbox[1]-new_bbox[1])//self.res)
            oh = min(old_map.shape[0],new_map.shape[0]-sy)
            ow = min(old_map.shape[1],new_map.shape[1]-sx)
            new_map[sy:sy+oh,sx:sx+ow] = old_map[:oh,:ow]
        return new_map

    #we keep in pcd because of normal calculations
    def get_preprocess_mask(self,points):
        up = [0,0,1]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        
        pcd.estimate_normals()
        pcd.normalize_normals()
        pcd.orient_normals_to_align_with_direction(up)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        height_mask = (points[:,2] >= self.min_height) & (points[:,2] <= self.max_height)
        #points = points[height_mask]
        #normals = normals[height_mask]

        #mask_normal = np.logical_not(np.logical_and((points[:,2] <= self.normal_threshold),(np.abs(np.dot(normals[:],up)) >= self.normal_threshold)))
        #print(np.abs(np.dot(normals[:],up)).tolist())
        
        
        normal_mask = np.logical_not(np.logical_and((np.abs(np.dot(normals[:],up)) >= self.normal_threshold),(points[:,2] <= self.normal_min_height)))
        #mask_normal = (np.abs(np.dot(normals[:],up)) >= self.normal_threshold)
        #points = points[normal_mask]

        return np.logical_and(height_mask,normal_mask)

    def generate_semantic_pcds(self,transform,segmentator):
        pcds = []
        ids = []

        for i in range(self.last_semantics,len(segmentator.static_instances_per_frame)):    
            for type_idx in segmentator.type_list:
                points = np.empty((0,3))
                for k in range(len(segmentator.static_instances_per_frame[i][type_idx])):
                    instance = segmentator.static_instances_per_frame[i][type_idx][k]
                    points = np.append(points,np.asarray(instance.pcd.points),0)
                if points.shape[0] != 0:
                    points = (np.insert(points,3,1,1)@transform.transpose())[:,:3]
                    if type_idx in ids:
                        index = ids.index(type_idx)
                        pcds[index] = np.append(pcds[index],points,0)
                    else:
                        pcds.append(points)
                        ids.append(type_idx)

        self.last_semantics = len(segmentator.static_instances_per_frame)


        return ids,pcds

    
    def add_pcd(self,points,semantic_ids,semantic_points):
        mask = self.get_preprocess_mask(points)
        #self.points_all = np.append(self.points_all,points,axis=0)
        #heights = points[:,2]
        
        start = time.time()

        points_rounded = np.squeeze(np.dstack(((points[:,0]-self.bbox[0])//self.res,(points[:,1]-self.bbox[1])//self.res)).astype(int))
        if len(points_rounded.shape) < 2:
            points_rounded = points_rounded[None]
        
        unique,counts = np.unique(points_rounded[mask],axis=0,return_counts=True)
        self.count[unique[:,1],unique[:,0]] += 1
        self.abs[unique[:,1],unique[:,0]] += counts[:]

        max_sid = np.argsort(points[:,2])
        min_sid = np.flip(max_sid,0)

        pr_mins = points_rounded[min_sid]
        pr_maxs = points_rounded[max_sid]

        minhm = np.full_like(self.min,np.nan)
        minhm[pr_mins[:,1],pr_mins[:,0]] = points[min_sid,2]
        self.min = np.fmin(self.min,minhm)

        maxhm = np.full_like(self.max,np.nan)
        maxhm[pr_maxs[:,1],pr_maxs[:,0]] = points[max_sid,2]
        self.max = np.fmax(self.max,maxhm)

        for id,pcd in zip(semantic_ids,semantic_points):
            pcd_rounded = np.squeeze(np.dstack(((pcd[:,0]-self.bbox[0])//self.res,(pcd[:,1]-self.bbox[1])//self.res)).astype(int))
            if len(pcd_rounded.shape) < 2:
                pcd_rounded = pcd_rounded[None]
            self.semantic[pcd_rounded[:,1],pcd_rounded[:,0]] = id

        end = time.time()
        self.cast_time_sum += end-start
        self.cast_time_n += 1

    def update(self,transform,points,segmentator):
        points = (np.insert(points,3,1,1)@transform.transpose())[:,:3]
        semantic_ids,semantic_points = self.generate_semantic_pcds(transform,segmentator)

        new_bbox = self.get_bbox(points)
        #print(new_bbox)
        nw = int((new_bbox[2]-new_bbox[0])//self.res)+1
        nh = int((new_bbox[3]-new_bbox[1])//self.res)+1
        #print(nh,nw)
        self.abs = self.expand_map(self.abs,new_bbox,nw,nh)
        self.count = self.expand_map(self.count,new_bbox,nw,nh,0)
        self.min = self.expand_map(self.min,new_bbox,nw,nh,np.nan)
        self.max = self.expand_map(self.max,new_bbox,nw,nh,np.nan)
        self.semantic = self.expand_map(self.semantic,new_bbox,nw,nh,np.nan)

        self.bbox = new_bbox
        self.w = nw
        self.h = nh
        
        self.add_pcd(points,semantic_ids,semantic_points)

        #new_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.points_all))
        #o3d.visualization.draw_geometries([new_pcd])

    def reset(self,transform,points,segmentator):
        self.abs = None
        self.count = None
        self.min = None
        self.max = None
        self.semantic = None
        self.last_semantics = 0
        self.bbox = np.array([float('inf'),float('inf'),float('-inf'),float('-inf')])
        self.w = 0
        self.h = 0

        self.update(transform,points,segmentator)

    def get_mean(self):
        mask = (self.count != 0)
        mean = np.zeros_like(self.abs)
        mean[mask] = np.divide(self.abs[mask],self.count[mask])
        return mean

#####def generate_iterative_map(path,name,aligned_vggt,floor_trans,skip,res,min_height=0.1,max_height=2.0,normal_min_height=0.2,normal_threshold=0.8,downsample_rate = 10):
#####    block_idx = np.reshape(np.arange(len(aligned_vggt.images)),(-1,aligned_vggt.block_size))
#####    block_idx = block_idx[int(block_idx.shape[0]*skip):]
#####    block_count = block_idx.shape[0]
#####    block_size = block_idx.shape[1]
#####    bbox = []
#####    sizes = []
#####
#####    aggregated_pcd = renderer.create_aggregated_pcd_offline(aligned_vggt,downsample_rate)
#####    aggregated_pcd.transform(floor_trans)
#####    #x_start,x_end,y_start,y_end,ny,nx = predict_2d_map_dimensions(aggregated_pcd,res)
#####    
#####    #parameters = [] #x_start,x_endy_start,y_end,res,ny,nx
#####    #parameters = np.array([x_start,x_end,y_start,y_end,res,ny,nx],dtype=np.float64)
#####    
#####    mean_map = []
#####    abs_map = []
#####    min_map = []
#####    max_map = []
#####    count_map = []
#####    #aggregated_abs_map = np.zeros((ni,ny,nx))
#####    #aggregated_min_map = np.zeros((ni,ny,nx))
#####    #aggregated_max_map = np.zeros((ni,ny,nx))
#####
#####    camera_positions = []
#####    camera_vectors = []
#####    for i in block_idx.flatten():
#####        pose = aligned_vggt.world_camera_poses[i]
#####    #for pose in aligned_vggt.world_camera_poses:
#####        camera_positions.append(floor_trans@np.append(pose[0:3,3],1))
#####        camera_vectors.append(floor_trans[0:3,0:3]@np.array(pose[0:3,2]))
#####    camera_positions = np.array(camera_positions)[:,0:2]
#####    camera_vectors = np.array(camera_vectors)[:,0:2]
#####    norms = np.linalg.norm(camera_vectors,axis=1)
#####    for i in range(norms.shape[0]):
#####        camera_vectors[i,:] = camera_vectors[i,:]/norms[i]
#####    camera_positions = camera_positions.reshape((-1,block_size,2)).tolist()
#####    camera_vectors = camera_vectors.reshape((-1,block_size,2)).tolist()
#####
#####    #np.array(aligned_vggt.world_camera_poses)[:,0:3,2]
#####    nav = NavMapGenerator(res,min_height=min_height,max_height=max_height,normal_min_height=normal_min_height,normal_threshold=normal_threshold)
#####
#####    for j in range(block_count):
#####        pcds = []
#####        for i in block_idx[j]:
#####            pcd = o3d.geometry.PointCloud()
#####            #img = torch.tensor(aligned_vggt.images[i]).permute(1,2,0).detach().cpu().numpy()[aligned_vggt.depth_masks[i]]
#####            points = aligned_vggt.generated_pcds[i][aligned_vggt.depth_masks[i]]
#####            pcd.points = o3d.utility.Vector3dVector(points.reshape(-1,3)[::downsample_rate])
#####            pcd.transform(floor_trans)
#####            pcds.append(pcd)
#####        #pcd.colors = o3d.utility.Vector3dVector(img.reshape(-1,3)[::downsample_rate])
#####        
#####        nav.update(pcds)
#####
#####        bbox.append(nav.bbox.tolist())
#####        sizes.append([nav.h,nav.w])
#####        #parameters.append([x_start,x_end,y_start,y_end,res,ny,nx])
#####        mean_map.append(nav.get_mean().tolist())
#####        abs_map.append(nav.abs.tolist())
#####        count_map.append(nav.count.tolist())
#####        min_map.append(nav.min.tolist())
#####        max_map.append(nav.max.tolist())
#####        #aggregated_min_map[i] = map2d_min
#####        #aggregated_max_map[i] = map2d_max
#####        print(f"Processed {j+1}/{block_count}")
#####    data = {}
#####
#####    #plt.imshow(nav.count)
#####
#####    data['block_count'] = block_count
#####    data['block_size'] = block_size
#####    data['res'] = res
#####    data['bbox'] = bbox
#####    data['shapes'] = sizes
#####    #data['param'] = parameters
#####    data['cam_pos'] = camera_positions
#####    data['cam_forward'] = camera_vectors
#####    data['mean'] = mean_map
#####    data['abs'] = abs_map
#####    data['min'] = min_map
#####    data['max'] = max_map
#####
#####    
#####    #o3d.io.write_point_cloud(os.path.join(path,name+"_pcd.ply"),aggregated_pcd)
#####    with open(os.path.join(path,name+"_map.json"), 'w') as f:
#####        json.dump(data, f)
#####        
#####    return data

def import_iterative_map(filename):

    #map_location = "./output/plane/tdk_test_map.json"

    with open(filename, 'r') as f:
            map_json = json.load(f)

    #map_json = data

    map = types.SimpleNamespace()
    map.block_size = map_json['block_size'] #scalar
    map.block_count = map_json['block_count'] #scalar
    map.res = map_json['res'] #scalar, resolution of each cell in meters
    map.bbox = np.array(map_json['bbox']) #shape (block_count:4), items: x_start, y_start, x_end, y_end
    map.shapes = np.array(map_json['shapes']) #shape (block_count:2), (h,w) of the map at n
    map.cam_forward = np.array(map_json['cam_forward']) #shape (block_count,block_size,2), x,y of a normalized vector facing the same way as the camera
    map.cam_pos = np.array(map_json['cam_pos']) #shape (block_count,block_size,2), x,y of camera positions in world coordinates
    map.mean = [] #an array of length block_count, containing the mean map (sum divided by frames seen) at each block
    map.abs = [] #an array of length block_count, containing the absolute map (sum of points in cell) at each block
    map.min = [] #an array of length block_count, containing the minimum map (min height in cell) at each block
    map.max = [] #an array of length block_count, containing the maximum map (max height in cell) at each block
    for i in range(len(map_json['abs'])):
        map.mean.append(np.array(map_json['mean'][i]))
        map.abs.append(np.array(map_json['abs'][i]))
        map.min.append(np.array(map_json['min'][i]))
        map.max.append(np.array(map_json['max'][i]))
    return map