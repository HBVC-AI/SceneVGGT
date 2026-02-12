import torch
import open3d as o3d
import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class Renderer:
    def __init__(self):
        self.colour_map = self.get_color_map()
    
    @staticmethod
    def get_color_map(N=256):
        """Generates the colors used by MiVOS, we use these to recognize the annotations
        the first color is the background which we don't need so we drop it
        """
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([b, g, r])

        return cmap[1:] # skip background color

    def create_aggregated_pcd(self,aligned_vggt, last_block = False,downsample_rate = 1):
        fully_aggregated_pcd = o3d.geometry.PointCloud()
        start_frame = 0
        if last_block:
            start_frame = len(aligned_vggt.images) - aligned_vggt.block_size
        for i in range(start_frame,len(aligned_vggt.images)):
            pcd = o3d.geometry.PointCloud()
            img = torch.tensor(aligned_vggt.images[i]).permute(1,2,0).detach().cpu().numpy()[aligned_vggt.depth_masks[i]]
            points = aligned_vggt.generated_pcds[i][aligned_vggt.depth_masks[i]]
            if points.shape[0] > 0:
                pcd.points = o3d.utility.Vector3dVector(points.reshape(-1,3)[::downsample_rate])
                pcd.colors = o3d.utility.Vector3dVector(img.reshape(-1,3)[::downsample_rate])
                fully_aggregated_pcd += pcd
        return fully_aggregated_pcd
    
    def create_aggregated_pcd_offline(self,aligned_vggt,downsample_rate = 1):
        fully_aggregated_pcd = o3d.geometry.PointCloud()
        for i in range(len(aligned_vggt.images)):
            pcd = o3d.geometry.PointCloud()
            img = torch.tensor(aligned_vggt.images[i]).permute(1,2,0).detach().cpu().numpy()[aligned_vggt.depth_masks[i]]
            points = aligned_vggt.generated_pcds[i][aligned_vggt.depth_masks[i]]
            pcd.points = o3d.utility.Vector3dVector(points.reshape(-1,3)[::downsample_rate])
            pcd.colors = o3d.utility.Vector3dVector(img.reshape(-1,3)[::downsample_rate])
            fully_aggregated_pcd += pcd
        return fully_aggregated_pcd

        
    def create_semantic_pcd(self,aligned_vggt, semantic_segmentator, masked = True, last_block = False, downsample_rate = 1):
        fully_aggregated_pcd = o3d.geometry.PointCloud()
        start_frame = 0
        if last_block:
            start_frame = len(aligned_vggt.images) - aligned_vggt.block_size
        if masked:
            for i in range(start_frame,len(aligned_vggt.images)):
                
                for type_idx in semantic_segmentator.type_list:
                    for k in range(len(semantic_segmentator.static_instances_per_frame[i][type_idx])):
                        instance = semantic_segmentator.static_instances_per_frame[i][type_idx][k]
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(instance.pcd.points)
                        pcd.paint_uniform_color((self.colour_map[type_idx][2]/255,self.colour_map[type_idx][1]/255,self.colour_map[type_idx][0]/255))
                        fully_aggregated_pcd += pcd
        else:
            for i in range(start_frame,len(aligned_vggt.images)):
                pcd = o3d.geometry.PointCloud()
                img = torch.tensor(aligned_vggt.images[i]).permute(1,2,0).detach().cpu().numpy()
                img = img*255
                img = img.astype(np.uint8)
                img = np.ascontiguousarray(img)
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                img = np.stack((img,)*3, axis=-1)
                aggregated_mask = np.zeros((semantic_segmentator.vggt_img_size[1],semantic_segmentator.vggt_img_size[0]),dtype=bool)
                for type_idx in semantic_segmentator.type_list:
                    for k in range(len(semantic_segmentator.static_instances_per_frame[i][type_idx])):
                        instance = semantic_segmentator.static_instances_per_frame[i][type_idx][k]
                        point_mask = np.bitwise_and(aligned_vggt.depth_masks[i], instance.mask)
                        img[point_mask,0] = int(self.colour_map[type_idx][2])
                        img[point_mask,1] = int(self.colour_map[type_idx][1])
                        img[point_mask,2] = int(self.colour_map[type_idx][0])
                        aggregated_mask += point_mask
                img = img.astype(np.float32)
                img = img/255
                if not masked:
                    aggregated_mask = np.ones((semantic_segmentator.vggt_img_size[1],semantic_segmentator.vggt_img_size[0]),dtype=bool)
                img = img[aligned_vggt.depth_masks[i]&aggregated_mask]
                points = aligned_vggt.generated_pcds[i][aligned_vggt.depth_masks[i]&aggregated_mask]
                pcd.points = o3d.utility.Vector3dVector(points.reshape(-1,3)[::downsample_rate])
                pcd.colors = o3d.utility.Vector3dVector(img.reshape(-1,3)[::downsample_rate])
                fully_aggregated_pcd += pcd
        return fully_aggregated_pcd
    def create_instance_tracked_pcd(self, aligned_vggt, semantic_segmentator, instance_tracker, masked=True, last_block=False, downsample_rate=1):
        fully_aggregated_pcd = o3d.geometry.PointCloud()
        start_frame = 0
        if last_block:
            start_frame = len(aligned_vggt.images) - aligned_vggt.block_size
        for i in range(start_frame, len(aligned_vggt.images)):
            pcd = o3d.geometry.PointCloud()
            img = torch.tensor(aligned_vggt.images[i]).permute(1, 2, 0).detach().cpu().numpy()
            img = (img * 255).astype(np.uint8)
            img = np.ascontiguousarray(img)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = np.stack((img,) * 3, axis=-1)
            for type_idx in semantic_segmentator.type_list:
                for track_id, v in instance_tracker.subtracklets[type_idx].items():
                    for k in range(len(v)):
                        frame_id, mask_id, valid = instance_tracker.subtracklets[type_idx][track_id][k]
                        if frame_id == i:
                            if valid:
                                img[instance_tracker.per_frame_instance_masks[type_idx][frame_id][mask_id-1],0] = int(self.colour_map[(type_idx+track_id)%255][2])
                                img[instance_tracker.per_frame_instance_masks[type_idx][frame_id][mask_id-1],1] = int(self.colour_map[(type_idx+track_id)%255][1])
                                img[instance_tracker.per_frame_instance_masks[type_idx][frame_id][mask_id-1],2] = int(self.colour_map[(type_idx+track_id)%255][0])
                            else:
                                img[instance_tracker.per_frame_instance_masks[type_idx][frame_id][mask_id-1],0] = 0
                                img[instance_tracker.per_frame_instance_masks[type_idx][frame_id][mask_id-1],1] = 0
                                img[instance_tracker.per_frame_instance_masks[type_idx][frame_id][mask_id-1],2] = 0
            img = img.astype(np.float32)
            img = img/255
            img = img[aligned_vggt.depth_masks[i]]
            points = aligned_vggt.generated_pcds[i][aligned_vggt.depth_masks[i]]
            pcd.points = o3d.utility.Vector3dVector(points.reshape(-1,3)[::10])
            pcd.colors = o3d.utility.Vector3dVector(img.reshape(-1,3)[::10])
            fully_aggregated_pcd += pcd
        return fully_aggregated_pcd
    
    def memory_instance_tracked_render(self, aligned_vggt, semantic_segmentator, instance_tracker, masked=True, last_block=False, downsample_rate=1):
        start_frame = 0
        if last_block:
            start_frame = len(aligned_vggt.images) - aligned_vggt.block_size
        f_img = torch.tensor(aligned_vggt.images[0]).permute(1, 2, 0).detach().cpu().numpy()
        xyz_chunks, rgb_chunks = [], []
        for i in range(start_frame, len(aligned_vggt.images)):
            img = np.ones_like(f_img,dtype=np.uint8) * 255
            for type_idx in semantic_segmentator.type_list:
                for track_id, v in instance_tracker.subtracklets[type_idx].items():
                    if track_id == 0:
                        continue
                    for k in range(len(v)):
                        frame_id, mask_id, valid = instance_tracker.subtracklets[type_idx][track_id][k]
                        if frame_id == i:
                            img[instance_tracker.per_frame_instance_masks[type_idx][frame_id][mask_id-1],0] = type_idx
                            img[instance_tracker.per_frame_instance_masks[type_idx][frame_id][mask_id-1],1] = track_id // 255
                            img[instance_tracker.per_frame_instance_masks[type_idx][frame_id][mask_id-1],2] = track_id % 255
            img = img[aligned_vggt.depth_masks[i]]
            points = aligned_vggt.generated_pcds[i][aligned_vggt.depth_masks[i]]
            points = points.reshape(-1,3)[::10]
            colors = img.reshape(-1,3)[::10]
            xyz_chunks.append(points)
            rgb_chunks.append(colors)
        full_xyz, full_rgb = np.concatenate(xyz_chunks, 0), np.concatenate(rgb_chunks, 0)
        mask = np.all(full_rgb == np.array([255, 255, 255], dtype=np.uint8), axis=1)
        background_xyz, background_rgb = full_xyz[mask], full_rgb[mask]
        foreground_xyz, foreground_rgb = full_xyz[~mask], full_rgb[~mask]
        return background_xyz,background_rgb,foreground_xyz,foreground_rgb

    def create_instance_tracked_images(self,aligned_vggt,instance_tracker,type_list, last_block = False, downsample_rate = 1):
        img_list = []
        start_frame = 0
        if last_block:
            start_frame = len(aligned_vggt.images) - aligned_vggt.block_size
        for i in range(start_frame,len(aligned_vggt.images)):
            img = torch.tensor(aligned_vggt.images[i]).permute(1,2,0).detach().cpu().numpy()
            img = img*255
            img = img.astype(np.uint8)
            img = np.ascontiguousarray(img)
            img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
            img_list.append(img)
        for type_idx in type_list:
            for i, v in instance_tracker.subtracklets[type_idx].items():
                for k in range(len(v)):
                    frame_id, mask_id, is_alive = instance_tracker.subtracklets[type_idx][i][k]
                    if not is_alive:
                        continue
                    col_idx = type_idx
                    if i == 0:
                        col_idx = 0
                    img_list[frame_id][instance_tracker.per_frame_instance_masks[type_idx][frame_id][mask_id-1],0] = int(self.colour_map[col_idx][0])
                    img_list[frame_id][instance_tracker.per_frame_instance_masks[type_idx][frame_id][mask_id-1],1] = int(self.colour_map[col_idx][1])
                    img_list[frame_id][instance_tracker.per_frame_instance_masks[type_idx][frame_id][mask_id-1],2] = int(self.colour_map[col_idx][2])
                    mask = instance_tracker.per_frame_instance_masks[type_idx][frame_id][mask_id-1]
                    coords = np.column_stack(np.where(mask))
                    if coords.size > 0:
                        y, x = coords.mean(axis=0).astype(int)
                        cv.putText(img_list[frame_id],
                            str(i),(x, y),cv.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255),1,cv.LINE_AA)
        return img_list

class Visualizer:
    def __init__(self,background_color = np.asarray([0,0,0]), point_size = 2.0):
        self.background_color = background_color
        self.point_size = point_size
        self.renderer = Renderer()
        
    def visualize(self, aligned_vggt, aggregated_pcd, render_pcd = True, render_trajectory = True):
        if aggregated_pcd is None and render_pcd == True:
            start_time = time.time()
            aggregated_pcd = self.renderer.create_aggregated_pcd(aligned_vggt,10)
            end_time = time.time()
            print(f"Aggregated {len(aligned_vggt.images)} frames in {end_time - start_time}s")
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis_options = vis.get_render_option()
        vis_options.background_color = np.asarray([0,0,0])
        vis_options.point_size = 2.0
        # Initiate interactive viewer
        vis.clear_geometries()
        if render_trajectory:
            pcd = o3d.geometry.PointCloud()
            points = []
            number_of_poses = len(aligned_vggt.world_camera_poses)
            cl_space = np.linspace(0, 1, number_of_poses+1)
            cmap = plt.cm.get_cmap("plasma")
            colour_map = cmap(cl_space)[:, :3]   
            for i in range(len(aligned_vggt.world_camera_poses)):
                points.append([aligned_vggt.world_camera_poses[i][0,3],aligned_vggt.world_camera_poses[i][1,3],aligned_vggt.world_camera_poses[i][2,3]])
                scale_factor = 0.02
                extent = [scale_factor,scale_factor,scale_factor]
                center = aligned_vggt.world_camera_poses[i][:,3]
                rotation = aligned_vggt.world_camera_poses[i][:,:3]
                obb = o3d.geometry.OrientedBoundingBox(center=center, R=rotation, extent=extent)
                obb.color = colour_map[i]
                vis.add_geometry(obb)
            points = np.asarray(points)
            pcd.points = o3d.utility.Vector3dVector(points.reshape(-1,3))
            pcd.paint_uniform_color((0.0,1.0,0.0))
            vis.add_geometry(pcd) # Camera centers
        if render_pcd:
            vis.add_geometry(aggregated_pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.run()
        vis.destroy_window()
