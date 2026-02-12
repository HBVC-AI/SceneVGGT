''' ANNA - Autonomous Node Navigational Agent

További nev alternatívak: 
    GUIDE - Goal-directed Unified Indoor Direction Engine
    AIDA - Adaptive Indoor Direction Assistant
    COMPASS - Contextual Object-aware Module for Pathfinding & Autonomous Spatial Support'''

import numpy as np

from modules.floor_plane import FloorPlaneFinder
from modules.navigation_new import NavMapGenerator
from modules.aligned_vggt_main_lidar import AlignedVGGTLidar
from modules.anna_vis import Visualization
from modules.anna_path import PathFinder
from modules.anna_heuristics import GoalHeuristic, FrontierExploration

class Anna():
    def __init__(
        self,
        vggt: AlignedVGGTLidar,
        obstacle_radius_m: float = 0.3,
        sample_step_m: float = 0.05,
        mean_threshold: float = 1.0,
        floor_height_threshold: float = 0.2,
    ):
        self.vggt = vggt
        self.obstacle_radius_m = obstacle_radius_m
        self.sample_step_m = sample_step_m
        self.mean_threshold = mean_threshold
        self.floor_height_threshold = floor_height_threshold

    
    # --- Get navigation data + helper ---
    def inflate_obstacles(
        self,
        obs_map: np.ndarray, 
        cell_res: float = 0.05, 
        obstacle_radius_m: float = 0.3,
        ) -> np.ndarray: 
    
        obs_map = obs_map.astype(bool, copy=False)
        r_cells = int(np.ceil(obstacle_radius_m / cell_res))    # how many cells correspond to the given radius

        if r_cells > 0:
            yy, xx = np.ogrid[-r_cells:r_cells+1, -r_cells:r_cells+1]
            disk = (xx*xx + yy*yy) <= (r_cells * r_cells)   # shape: (2*r_cells+1, 2*r_cells+1)

            # Paddding to get a circle-like effect at the edges
            pad = r_cells
            padded = np.pad(obs_map, pad, mode='constant', constant_values=False)

            # Sliding window view (H, W, k, k), k = 2*r_cells+1
            from numpy.lib.stride_tricks import sliding_window_view
            win = sliding_window_view(padded, disk.shape)   # shape: (H, W, k, k)

            # Any cell True under the circular mask → center is also an obstacle
            inflated = np.any(win & disk, axis=(-2, -1))    # shape: (H, W)

            return inflated
        else:
            return obs_map
        
    def list_seen_classes(slef, semantic_map: np.ndarray):
        return np.unique(semantic_map[np.isfinite(semantic_map)])

    def get_nav_data(
        self,
        block_idx: int, 
        navmap: NavMapGenerator,
        plane: FloorPlaneFinder
    ) -> dict:
        output = dict()
        cam_pos = []
        pose = self.vggt.world_camera_poses[block_idx]
        cam_pos.append(plane.transform@np.append(pose[0:3,3],1))
        cam_pos = np.array(cam_pos)[:,0:2]
        mean_map = np.array(navmap.get_mean())
        max_map = np.array(navmap.max)
        min_map = np.array(navmap.min)
        sem_map = np.array(navmap.semantic)
        res = navmap.res

        known = np.isfinite(min_map)      # where we have any data
        holes = ~known                          # areas where we have no data -> obstacles but no inflation
        obs_seed = np.isfinite(mean_map) & (mean_map > self.mean_threshold) # where we have enough data to consider as obstacle -> to be inflated
        obs_inflated = self.inflate_obstacles(
            obs_map = obs_seed, 
            cell_res=res, 
            obstacle_radius_m=self.obstacle_radius_m)

        base_map = known & (min_map < self.floor_height_threshold)    # where points are known to be low enough -> probably floor

        curr_map = mean_map.astype(np.float32).copy()   # map where can see where are obstacles / holes
        curr_map[holes] = np.nan

        output["mean"] = mean_map
        output["max"] = max_map
        output["min"] = min_map
        output["semantic"] = sem_map
        output["cam_pos"] = np.array(cam_pos)
        output["res"] = res
        output["bbox"] = np.array(navmap.bbox)
        output["base_bin_map"] = base_map
        output["holes"] = holes
        output["obs_bin_map"] = obs_inflated | holes
        output["curr_map"] = curr_map

        seen_classes = self.list_seen_classes(sem_map)
        print("Seen classes:", seen_classes)

        return output


    ########################
    # --- Main function ---#
    ########################
    def get_navigation_and_visualize(
        self, 
        block_idx: int, 
        navmap: NavMapGenerator,
        plane: FloorPlaneFinder,
        goal_id: int = 1,  # chair = 56, bicycle = 1
    ):
        nav_data: dict = {}
        path_pts: np.ndarray
        nav = PathFinder()
        vis = Visualization()
        heur = GoalHeuristic()
        exp = FrontierExploration()
        goal_point: np.array = []

        nav_data = self.get_nav_data(
            block_idx=block_idx, 
            navmap=navmap,
            plane=plane
        )

        regions = heur.extract_goal_regions(
            sem_map=nav_data.get("semantic"),
            class_id=goal_id,
            res=nav_data.get("res"),
            bbox=nav_data.get("bbox"),
            min_cells=10,                 # around 10-20
            closing_radius_cells=1,
            connectivity=8,
        )

        start_rc = nav_data.get("cam_pos")[-1]
        print("Start position (world x,y):", start_rc)
        
        goal_region = heur.pick_nearest_region(regions, start_rc)

        if goal_region is None:
            print("No valid goal region found for class", goal_id, " We looking for frontier subgoal!")
            sub_goal = exp.get_frontier_subgoal(nav_data=nav_data)

            if sub_goal is None:
                print("PROBLEM: No reachable frontier and no valid goal. Probably there is no item you looking for.")
                goal_point = start_rc
            else:
                goal_point = sub_goal
        else:
            print("Chosen goal region area:", goal_region.area_cells)
            print("Goal centroid grid (r,c):", goal_region.centroid_rc)
            print("Goal centroid world (x,y):", goal_region.centroid_xy)
            goal_point = goal_region.centroid_xy
        
        path_pts = nav.plan_path(
            obs_map=nav_data.get("obs_bin_map"),
            start_xy=start_rc,
            goal_point=goal_point,
            cell_res=nav_data.get("res"),
            bbox=nav_data.get("bbox"),
            sample_step_m=0.5
        )
        
        if path_pts.size == 0:
            print("No feasible path found.")
        else:
            print(f"Path found with {path_pts.shape[0]} points.")

        vis.draw_map_with_path_cv2(
            curr_map=nav_data.get("max"),
            bbox=nav_data.get("bbox"),
            cell_res=nav_data.get("res"),
            path_pts=path_pts,
            goal_point=goal_point,
            start_point=start_rc,
            sem_map=nav_data.get("semantic"),
            highlight_class=goal_id,
            block_idx = block_idx
        )