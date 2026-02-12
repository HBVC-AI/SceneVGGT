import torch
import open3d as o3d
import time
import numpy as np
from vggt.models.vggt import VGGT
from modules.aligned_vggt_main_lidar import AlignedVGGTLidar
from modules.input_stream_main import InputStreamMain
from modules.visualizer import *
from modules.semantic_segmentator import SemanticSegmentator
from modules.instance_tracker import InstanceTracker
from modules.static_object_manager import *
from ultralytics import YOLO
from tqdm import tqdm
import argparse
import os
def main(use_semantics = True, use_instance_tracking = True, visualize = True, save_vis = True, 
         vis_path=f"output/new_scale_agg.ply", num_of_imgs_to_use = -1, step =1,block_size = 10,
         keyframe_per_block = 10, depth_conf_threshold = 1.1, semantic_conf_threshold = 0.1,memory_decay=0.1,
         rgb_folder=None,rgb_files=None,rgb_pattern="*.png",depth_folder=None,depth_files=None,
         depth_pattern="*.png",depth_scaling=1000.0,
         lidar_far_plane=5.0,save_renderer_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading VGGT model using device: {device}")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    input_stream = InputStreamMain(rgb_folder=rgb_folder,rgb_pattern=rgb_pattern,depth_folder=depth_folder,depth_pattern=depth_pattern,depth_scaling=depth_scaling,data_limit=num_of_imgs_to_use,step=step)
    w, h = input_stream.img_size
    fixed_w = 518
    vggt_img_size = (fixed_w, int(round(h * (fixed_w / w) / 14) * 14))
    if visualize == True:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis_options = vis.get_render_option()
        ctr = vis.get_view_control()
        ctr.rotate(90.0, 90.0, 0.0)
        ctr.translate(-10.0, 0.0)
        vis_options.background_color = np.asarray([0,0,0])
        vis_options.point_size = 1.0
        # Initiate interactive viewer
        vis.clear_geometries()
        vis.poll_events()
        vis.update_renderer()
        if save_renderer_path is not None:
            os.makedirs(save_renderer_path,exist_ok=True)
    if use_semantics or use_instance_tracking:
        yolo_model = YOLO("yolov9e-seg.pt")
        type_list = list(range(80))
        yolo_model.overrides["classes"] = type_list
        type_name_dict = { 0: "person",  1: "bicycle",  2: "car",  3: "motorcycle",  4: "airplane",  5: "bus",  6: "train",  7: "truck",  8: "boat",  9: "traffic light",  10: "fire hydrant",  11: "stop sign",  12: "parking meter",  13: "bench",  14: "bird",  15: "cat",  16: "dog",  17: "horse",  18: "sheep",  19: "cow",  20: "elephant",  21: "bear",  22: "zebra",  23: "giraffe",  24: "backpack",  25: "umbrella",  26: "handbag",  27: "tie",  28: "suitcase",  29: "frisbee",  30: "skis",  31: "snowboard",  32: "sports ball",  33: "kite",  34: "baseball bat",  35: "baseball glove",  36: "skateboard",  37: "surfboard",  38: "tennis racket",  39: "bottle",  40: "wine glass",  41: "cup",  42: "fork",  43: "knife",  44: "spoon",  45: "bowl", 46: "banana",  47: "apple",  48: "sandwich",  49: "orange",  50: "broccoli",  51: "carrot",  52: "hot dog",  53: "pizza",  54: "donut",  55: "cake",  56: "chair",  57: "couch",  58: "potted plant",  59: "bed",  60: "dining table",  61: "toilet",  62: "tv",  63: "laptop",  64: "mouse",  65: "remote",  66: "keyboard",  67: "cell phone",  68: "microwave",  69: "oven",  70: "toaster",  71: "sink",  72: "refrigerator",  73: "book",  74: "clock",  75: "vase",  76: "scissors",  77: "teddy bear",  78: "hair drier",  79: "toothbrush"}
    if use_semantics or use_instance_tracking:
        semantic_segmentator = SemanticSegmentator(yolo_model,type_name_dict,type_list,vggt_img_size,conf_threshold=semantic_conf_threshold)
        instance_tracker = InstanceTracker(vggt_img_size,type_list)
        static_object_manager = StaticObjectManager(type_list,vggt_img_size,memory_decay)
    renderer = Renderer()
    alignment_time = 0.0
    semantic_time = 0.0
    instance_time = 0.0
    object_mem_time = 0.0
    total_time = 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    aligned_vggt = AlignedVGGTLidar(model,block_size = block_size,keyframe_per_block = keyframe_per_block,
                                    keyframe_memory_cap = keyframe_per_block, 
                                    depth_conf_threshold = depth_conf_threshold,
                                    device = device,
                                    far_plane = lidar_far_plane)
    rgb_map = {}
    rgb_rng = np.random.default_rng(1234)
    last_full_block = list(range(0,len(input_stream),aligned_vggt.block_size))[-1] # To handle if the frame num is not divisible by the block size
    if len(input_stream) % aligned_vggt.block_size != 0:
        last_full_block -= aligned_vggt.block_size
    for block_idx in tqdm(range(0,last_full_block+1,aligned_vggt.block_size)):
        aggregated_pcd = o3d.geometry.PointCloud()
        
        # Get frames relevant to this block
        frames = input_stream.get_frames(aligned_vggt.block_size)
        if use_instance_tracking == True:
            block_indices = list(range(block_idx,int(np.amin([block_idx+aligned_vggt.block_size,len(input_stream)]))))
            tracked_frame_indices = aligned_vggt.keyframe_list_idx + block_indices
        if frames[0] is None: 
            break
        # Process new frames and keyframes with VGGT
        print(f"Block {block_idx}")
        start_time = time.time()
        _, _, _ = aligned_vggt.process_frames(frames, block_idx)
        end_time = time.time()
        print(f"Processed and aligned {len(aligned_vggt.keyframe_list_idx) + len(frames)} frames in {end_time - start_time}s")
        alignment_time += end_time - start_time
        total_time += end_time - start_time
        if use_semantics == True:
            start_time = time.time()
            semantic_segmentator.segment_frames(frames,aligned_vggt,block_idx)
            end_time = time.time()
            print(f"Extracted static instances for {len(aligned_vggt.keyframe_list_idx) + len(frames)} frames in {end_time - start_time}s")
            semantic_time += end_time - start_time
            total_time += end_time - start_time
        if use_instance_tracking == True:
            start_time = time.time()
            instance_tracker.run_instance_tracking(aligned_vggt,semantic_segmentator,tracked_frame_indices)
            end_time = time.time()
            instance_time += end_time - start_time
            total_time += end_time - start_time
            print(f"Performed instance tracking for {len(aligned_vggt.keyframe_list_idx) + len(frames)} frames in {end_time - start_time}s")
            start_time = time.time()
            bg_memory_xyz,bg_memory_rgb,fg_memory_xyz,fg_memory_rgb = renderer.memory_instance_tracked_render(aligned_vggt,semantic_segmentator, instance_tracker, False,True,10)
            static_object_manager.fg_memory_xyz = np.vstack([static_object_manager.fg_memory_xyz,fg_memory_xyz]) if static_object_manager.fg_memory_xyz.size else fg_memory_xyz
            static_object_manager.fg_memory_rgb = np.vstack([static_object_manager.fg_memory_rgb,fg_memory_rgb],dtype=np.uint8) if static_object_manager.fg_memory_rgb.size else fg_memory_rgb
            static_object_manager.bg_memory_xyz = np.vstack([static_object_manager.bg_memory_xyz,bg_memory_xyz]) if static_object_manager.bg_memory_xyz.size else bg_memory_xyz
            static_object_manager.bg_memory_rgb = np.vstack([static_object_manager.bg_memory_rgb,bg_memory_rgb],dtype=np.uint8) if static_object_manager.bg_memory_rgb.size else bg_memory_rgb
            instance_tracker = static_object_manager.update_objects(aligned_vggt,instance_tracker,semantic_segmentator,tracked_frame_indices)
            end_time = time.time()
            object_mem_time += end_time - start_time
            total_time += end_time - start_time
            print(f"Updated static objects frames in {end_time - start_time}s")
        aligned_vggt.clean_memory()
        start_time = time.time()
        if visualize == True:
            if use_instance_tracking == True:
                rendered_pcd = o3d.geometry.PointCloud()
                rendered_pcd.points = o3d.utility.Vector3dVector(static_object_manager.fg_memory_xyz)
                xyz = np.asarray(static_object_manager.fg_memory_xyz)
                rgb = np.asarray(static_object_manager.fg_memory_rgb)
                if xyz.size == 0 or rgb.size == 0:
                    pass
                else:
                    if rgb.ndim == 1:
                        rgb = rgb.reshape(-1, 3)

                    n = min(len(xyz), len(rgb))
                    xyz = xyz[:n]
                    rgb = rgb[:n]

                    rendered_pcd = o3d.geometry.PointCloud()
                    rendered_pcd.points = o3d.utility.Vector3dVector(xyz)

                    u, inv = np.unique(rgb, axis=0, return_inverse=True)

                    for k in map(tuple, u):
                        if k not in rgb_map:
                            rgb_map[k] = rgb_rng.integers(64, 256, size=3, dtype=np.uint8)

                    lut = np.array([rgb_map[tuple(k)] for k in u], dtype=np.uint8)
                    rendered_pcd.colors = o3d.utility.Vector3dVector(lut[inv].astype(np.float32) / 255.0)
                vis.clear_geometries()
            elif use_semantics == True:
                rendered_pcd = renderer.create_semantic_pcd(aligned_vggt,semantic_segmentator,False,True,downsample_rate=10)
            else:
                rendered_pcd = renderer.create_aggregated_pcd(aligned_vggt,True,downsample_rate=10)
            R = rendered_pcd .get_rotation_matrix_from_axis_angle([np.radians(220), 0, 0])
            rendered_pcd .rotate(R, center=(0, 0, 0)) 
            if len(rendered_pcd.points) > 0:
                vis.add_geometry(rendered_pcd)
                vis.poll_events()
                vis.update_renderer()
                if save_renderer_path is not None:
                    img = vis.capture_screen_float_buffer(do_render=True)
                    img_np = np.asarray(img)*255
                    cv.imwrite(save_renderer_path+f"{block_idx:04d}.png", cv.cvtColor(img_np.astype(np.uint8),cv.COLOR_RGB2BGR))
            end_time = time.time()
            print(f"Rendered {len(aligned_vggt.keyframe_list_idx) + len(frames)} frames in {end_time - start_time}s")
    print(f"Total alignment time: {alignment_time}s ~ {len(input_stream) / alignment_time} FPS")
    if use_semantics:
        print(f"Total semantic time: {semantic_time}s ~ {len(input_stream) / semantic_time} FPS")
    if use_instance_tracking:
        print(f"Total instance time: {instance_time}s ~ {len(input_stream) / instance_time} FPS")
        print(f"Total object management time: {object_mem_time}s ~ {len(input_stream) / object_mem_time} FPS")
    print(f"Combined time: {total_time}s ~ {len(input_stream) / total_time} FPS")
    if visualize:
        vis.run()
        vis.destroy_window()
    if save_vis:
        print(f"Saving PCD...")
        aggregated_pcd = renderer.create_aggregated_pcd(aligned_vggt,False,downsample_rate=10)
        o3d.io.write_point_cloud(vis_path, aggregated_pcd)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--use-semantics", dest="use_semantics", action="store_true")
    parser.add_argument("--use-instance-tracking", dest="use_instance_tracking", action="store_true")
    parser.add_argument("--visualize", dest="visualize", action="store_true")
    parser.add_argument("--save-vis", dest="save_vis", action="store_true")
    parser.add_argument("--save-poses", dest="save_poses", action="store_true")
    parser.set_defaults(use_semantics=False, use_instance_tracking=False, visualize=False, save_vis=False)

    parser.add_argument("--num-of-imgs-to-use", type=int, default=-1)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--keyframe-per-block", type=int, default=10)
    parser.add_argument("--depth-conf-threshold", type=float, default=1.1)
    parser.add_argument("--semantic-conf-threshold", type=float, default=0.1)
    parser.add_argument("--memory-decay", type=float, default=0.0)
    parser.add_argument("--depth-scaling", type=float, default=1000.0)

    parser.add_argument("--rgb-folder", default=None)
    parser.add_argument("--rgb-pattern", default="*.png")
    parser.add_argument("--depth-folder", default=None)
    parser.add_argument("--depth-pattern", default="*.png")
    parser.add_argument("--rgb-files", nargs="*", default=None)
    parser.add_argument("--depth-files", nargs="*", default=None)

    parser.add_argument("--lidar-far-plane", type=float, default=10.0)

    parser.add_argument("--vis-path", default=f"output/new_scale_agg.ply")
    parser.add_argument("--save-renderer-path", default=None)
    args = parser.parse_args()
    main(
        use_semantics=args.use_semantics,
        use_instance_tracking=args.use_instance_tracking,
        visualize=args.visualize,
        save_vis=args.save_vis,
        vis_path=args.vis_path,
        num_of_imgs_to_use=args.num_of_imgs_to_use,
        step=args.step,
        block_size=args.block_size,
        keyframe_per_block=args.keyframe_per_block,
        depth_conf_threshold=args.depth_conf_threshold,
        semantic_conf_threshold=args.semantic_conf_threshold,
        memory_decay=args.memory_decay,
        rgb_folder=args.rgb_folder,
        rgb_files=args.rgb_files,
        rgb_pattern=args.rgb_pattern,
        depth_folder=args.depth_folder,
        depth_files=args.depth_files,
        depth_pattern=args.depth_pattern,
        depth_scaling=args.depth_scaling,
        lidar_far_plane=args.lidar_far_plane,
        save_renderer_path=args.save_renderer_path
    )