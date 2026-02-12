import torch
from vggt.models.vggt import VGGT
import open3d as o3d
import time
import numpy as np
from .utils import *
import copy
import concurrent.futures


def ransac(points,it_n,max_dist,min_cross_tr=0.0001,seed=1234):
    best_eq = None
    best_inliers = -1
    rnd = np.random.default_rng(seed)
    for i in range(it_n):
        samples = rnd.choice(points,3,replace=False,shuffle=True)

        n = np.cross(samples[1]-samples[0],samples[2]-samples[0])
        ln = np.linalg.norm(n)
        if ln < min_cross_tr:
            continue
        n /= ln
        eq = np.append(n,-1*np.dot(n,samples[0]))

        inliers = np.sum(np.abs(np.dot(points,eq[:3])+eq[3]) <= max_dist)
        if inliers > best_inliers:
            best_inliers = inliers
            best_eq = eq

    return best_eq

class FloorPlaneFinder:

    STATE_NO_UPDATE = 0 #plane is not updated
    STATE_REORIENT_TRANSFORM = 1 #plane is updated, but rotation remains, goal doesn't need to be updated
    STATE_RECALCULATE = 2 #plane changed and other modules need to recalculate

    def __init__(self):

        
    
        #updating memory
        self.transform = None #this is the currently accepted one
        self.inv_transform = None #inverse used for rendering
        self.transforms = [] #these are all calculated transforms
        self.eq = None #plane equations
        self.eqs = []
        
        
        #input data
        self.pcd = np.empty([0,3], dtype=np.float64) #points, stored as np array
        self.poses = np.empty([0,3,4], dtype=np.float64) #camera poses
        self.center = None #scene center used for transformation
        self.aligner_start = 0 #tracks the last frames added
        
        #parameters
        self.frame_step = 3 #only add points from every x frames
        self.dilation = 3 #dilate masks to when removing points
        self.ds_rate = 50 #downsample rate of points added each frame
        self.it_n = 2000 #number of iterations for pyransac3d
        self.max_dist = 0.05 #max inlier distance for pyransac3d
        error_tr_deg = 5 #max difference in degree between normals, if above, we consider them to represent different planes
        self.error_tr = 1-np.cos(error_tr_deg/180.0*np.pi) #same as error_tr_deg, but in cosine, for dot product testing
        self.d_tr = 0.1 #if the normals match, but the d coefficient difference is larger than this, we consider the two planes different (different height)
        self.check_last = 5 #we consider the last n transforms to decide which one to keep (needs to be odd)
        
        reorient_tr_deg = 10 #if the angle difference between plane normals are smaller than this, then we try to salve by rotation, otherwise we need to recalculate
        self.reorient_tr = np.cos(reorient_tr_deg/180.0*np.pi)
        self.reorient_transform = None
        
        #async exec
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=1) #creates processes
        self.async_eq = None #stores pyransac3d result

        #plane adjustment
        self.exp_d_adjust = False
        self.d_search_r = self.max_dist
        self.d_search_res = self.d_search_r/10
    
    def adjust_plane(self,eq):
        
        if self.center is not None and (np.dot(eq[:3],self.center)+eq[3]) < 0: #this means that the plane is flipped, we need to fix it
            eq *= -1
            #print("Needed to flip normal (in adjust)")
        
        deltas = []
        densities = []
        for d in np.arange(-self.d_search_r,self.d_search_r+self.d_search_res/2,self.d_search_res,dtype=np.float64):
            new_eq = eq.copy()
            new_eq[3] += d
            inliers_count = np.sum(np.abs(np.dot(self.pcd,new_eq[:3]) + new_eq[3]) < self.d_search_res)
            deltas.append(d)
            densities.append(inliers_count)
            #print(f"Offset: {d}, inlier count: {inliers_count}")

        max_delta = deltas[np.argmax(densities)]
        new_eq = eq.copy()
        new_eq[3] += max_delta

        return new_eq

    def add_points(self,aligner,segmentator,start=None):
        
        start_frame = start or self.aligner_start

        for i in range(start_frame,len(aligner.generated_pcds),self.frame_step):
            points = aligner.generated_pcds[i]
            h = points.shape[0]
            w = points.shape[1]
            mask = np.full((h,w),False)
            #print(points.reshape((-1,3)).shape)
            for _,v in segmentator.static_instances_per_frame[i].items(): #we filter semantics because floor is not included
                for instance in v:
                    mask = np.logical_or(mask,instance.mask)
            #print(np.sum(mask))
            mask = mask.astype(np.uint8)
            if self.dilation > 0:
                mask = cv.dilate(mask,np.ones((self.dilation*2+1,self.dilation*2+1)))
            mask = np.logical_not(mask)
            self.pcd = np.append(self.pcd,points[mask][::self.ds_rate],0)
        
        for i in range(start_frame,len(aligner.world_camera_poses)):
            poses = aligner.world_camera_poses[i:]
            self.poses = np.insert(self.poses,self.poses.shape[0],(poses),axis=0)
        
        if self.center is None:
            self.center = self.poses[0,0:3,3]
        self.aligner_start = len(aligner.generated_pcds)

    @staticmethod
    def plane_to_transform(peq,center):
        eq = np.array(peq).copy()
        def normalize(vec):
            return vec/np.linalg.norm(vec)

        #normal vector can be read from equation
        n = normalize(eq[0:3])

        #finding 2 points on the plane
        c_ind = np.argmax(n)
        n_coeff = np.delete(n,c_ind)
        c_coeff = eq[c_ind]
        d = eq[3]
        base1 = np.array([0,1],dtype=np.double)
        base2 = np.array([1,0],dtype=np.double)
        c1 = -1*(d+np.dot(base1,n_coeff))/c_coeff
        c2 = -1*(d+np.dot(base2,n_coeff))/c_coeff
        p1 = np.insert(base1,c_ind,c1)
        p2 = np.insert(base2,c_ind,c2)
        
        #x vector is from these 2 points
        vx = normalize(p1-p2)

        vt = normalize(center-p1)
        if np.dot(vt,n) < 0: #this means that the plane is flipped, we need to fix it
            n *= -1
            eq *= -1
            #print("Needed to flip normal")
        
        #y vector is from the cross product of the normal and x, order is important because of right hand rule
        vy = normalize(np.cross(n,vx))

        epsilon = 1**-7
        #if not (abs(np.dot(n,vx)) < epsilon and abs(np.dot(n,vy)) < epsilon and abs(np.dot(vx,vy)) < epsilon):
        #    print("BASIS VECTORS ARE NOT ORTHOGONAL!")

        #calculating the new origin, which will be the center projected to the plane
        pl = np.dot(n*-1,p1-center)
        new_origin = center-(n*pl)
        
        #creating transformation matrix
        basis = np.column_stack((vx, vy, n))
        trans = np.linalg.inv(basis)
        T = np.eye(4)
        T[0:3,0:3] = trans
        T[0:3,3] -= trans@new_origin
        return eq,T

    def atomic_fit(self):
        
        if self.center is None:
            self.center = np.array([0,0,0],dtype=np.float64)
        
        
        #I am going to ignore this for now, as we have no basis for assumed inlier prior
        #inlier_ratior = 0.1
        #n = lower_points_ds.shape[0]
        #pn = math.ceil(n*inlier_ratior)
        #Pe = (pn*(pn-1)*(pn-2))/(n*(n-1)*(n-2))
        #desired = 0.99
        #nit = math.ceil(math.log(1-desired)/math.log(1-Pe))
        
        eq = ransac(self.pcd,self.it_n,self.max_dist)
        if self.exp_d_adjust:
            eq = self.adjust_plane(eq)
        eq2,transform = self.plane_to_transform(eq,self.center)
        return eq2,transform

    def async_fit(self):        
        self.async_eq = self.executor.submit(ransac,self.pcd,self.it_n,self.max_dist)

    def collect_async(self):
        if self.async_eq is None or not self.async_eq.done():
            return None
        else:
            if self.center is None:
                self.center = np.array([0,0,0],dtype=np.float64)

            eq = self.async_eq.result()
            if self.exp_d_adjust:
                #print(f"PREEQ: {eq}")
                eq = self.adjust_plane(eq)
                #print(f"POSTEQ: {eq}")
            eq2,transform = self.plane_to_transform(eq,self.center)
            self.async_eq = None
            return eq2,transform
    
    def update_inv_transform(self):
        Rinv = np.transpose(self.transform[:3,:3])
        uinv = -1*(Rinv@self.transform[:3,3])
        self.inv_transform = np.eye(4)
        self.inv_transform[:3,:3] = Rinv
        self.inv_transform[:3,3] = uinv

    def update_plane(self,new_eq,new_transform):

        if np.dot(new_eq[:3],self.eq[:3]) > self.reorient_tr:
            #plan, we take (1,0,0) on the current plane, and rotate the new one, so that the (1,0,0) points in the same direction
            refv = np.array([1,0,0,1])

            #We have translation too, so have to do this instead of simple inversion
            Rinv = np.transpose(self.transform[:3,:3])
            uinv = -1*(Rinv@self.transform[:3,3])
            Tinv = np.eye(4)
            Tinv[:3,:3] = Rinv
            Tinv[:3,3] = uinv


            self.reorient_transform = new_transform@Tinv
            newv = self.reorient_transform@refv
            alpha = -1*np.arctan2(newv[1],newv[0]) #(1,0) has an angle of 0, so we rotate it in the other direction by the angle of the new vector
            R = np.eye(4)
            R[0,0] = np.cos(alpha)
            R[1,1] = np.cos(alpha)
            R[0,1] = -1*np.sin(alpha)
            R[1,0] = np.sin(alpha)
            new_transform = R@new_transform
            self.reorient_transform = R@self.reorient_transform

            self.transform = new_transform
            self.eq = new_eq
            self.update_inv_transform()

            return self.STATE_REORIENT_TRANSFORM
        
        self.transform = new_transform
        self.eq = new_eq
        self.update_inv_transform()

        return self.STATE_RECALCULATE

    def advance(self,aligner,segmentator): #return value is True if self.transform is updated
        
        if self.transform is None: #first run, fit and update
            self.add_points(aligner,segmentator)
            self.eq,self.transform = self.atomic_fit()
            self.update_inv_transform()
            #self.wup = np.transpose(self.transform[:3,:3]) @ self.up

            self.transforms.append(self.transform)
            self.eqs.append(self.eq)
            #self.wups.append(self.wup)
            
            return self.STATE_RECALCULATE
        
        if self.async_eq is None: #No async fitting done yet, run it now
            self.add_points(aligner,segmentator)
            self.async_fit()
            return self.STATE_NO_UPDATE
        
        async_res = self.collect_async() #Async run not done yet
        if async_res is None:
            return self.STATE_NO_UPDATE
        
        #if we are here, we have an async results, so we check if it's better
        current_eq = async_res[0]
        current_transform = async_res[1]
        
        self.transforms.append(current_transform)
        self.eqs.append(current_eq)

        updated = self.STATE_NO_UPDATE
        

        if (1-np.dot(current_eq[:3],self.eq[:3]) > self.error_tr) or (abs(self.eq[3]-current_eq[3]) > self.d_tr):
            to_check = self.eqs[-self.check_last-1:-1]
            score = 0
            for tceq in to_check: #for this, we check if it is parallel with most of the last n transforms
                error = 1-np.dot(tceq[:3],current_eq[:3])
                if (error > self.error_tr) or (abs(tceq[3]-current_eq[3]) > self.d_tr):
                    score -= 1
                else:
                    score += 1
            if score > 0: #if so, we update it
                updated = self.update_plane(current_eq,current_transform)
                
        #start new async process
        self.add_points(aligner,segmentator)
        self.async_fit()
        return updated
