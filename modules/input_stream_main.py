import os
import cv2
import csv
import numpy as np
import re
import glob
class InputStream:
    def next_frame(self):
        return None,None
    def get_frames(self,frame_cnt):
        return [None] * frame_cnt
    def eof(self):
        return True
    def __len__(self):
        return 0

class InputStreamMain(InputStream):
    def __init__(self,rgb_folder=None,rgb_files=None,rgb_pattern="*.png",depth_folder=None,depth_files=None,depth_pattern="*.csv",depth_scaling=1.0,timeout=10,data_limit=-1,step=1):
        assert not (rgb_folder is None and rgb_files is None), "Please specify an RGB source"
        self.mode = 0
        self.data_limit = data_limit
        self.step = step
        def natural_num_key(p):
            name = os.path.basename(p)
            nums = re.findall(r"\d+", name)
            key = tuple(int(n) for n in nums)
            return key if nums else (float("inf"), name)
        if rgb_folder is not None:
            self.img_files = sorted(glob.glob(os.path.join(rgb_folder, rgb_pattern)), key=natural_num_key)
        else:
            self.img_files = rgb_files

        img = cv2.cvtColor(cv2.imread(self.img_files[0]),cv2.COLOR_BGR2RGB)
        self.img_size = (img.shape[1],img.shape[0])
        self.depth_scaling = depth_scaling
        self.depth_type = depth_pattern.split('*')[-1]
        if depth_folder is not None:
            self.depth_files = sorted(glob.glob(os.path.join(depth_folder, depth_pattern)), key=natural_num_key)
        else:
            self.depth_files = depth_files
        if data_limit != -1:
            self.img_files = self.img_files[:data_limit:step]
            if self.depth_files is not None:
                self.depth_files = self.depth_files[:data_limit:step]
        else:
            self.img_files = self.img_files[::step]
            if self.depth_files is not None:
                self.depth_files = self.depth_files[::step]
        self.index = 0
        self.stream_thread = None

    def next_frame(self):
        if self.eof():
            return None,None, None
        img = cv2.cvtColor(cv2.imread(self.img_files[self.index]),cv2.COLOR_BGR2RGB)
        depthmat = None
        if self.depth_files is not None:
            if self.depth_type == ".csv":
                with open(self.depth_files[self.index], newline='') as csvfile:
                    csv_reader = csv.reader(csvfile, delimiter=',')
                    depthmat = np.array( [row for row in csv_reader], dtype=float) / self.depth_scaling
            else:
                depthmat = cv2.imread(self.depth_files[self.index],-1) / self.depth_scaling
                
        self.index += 1
        return img,depthmat, self.index-1
    def get_frames_uniform(self,frame_cnt):
        frames = []
        for i in range(frame_cnt):
            frame = self.next_frame()
            if frame[0] is None:
                break
            frames.append(frame)
        return frames
    
    def get_frames(self,frame_cnt):
        if self.eof():
            return [(None,None,None)]
        frames = []
        for i in range(frame_cnt):
            frame = self.next_frame()
            if frame[0] is None:
                break
            frames.append(frame)
        return frames
    
    def eof(self):
        return (self.index >= len(self.img_files))
    
    def __len__(self):
        return len(self.img_files)