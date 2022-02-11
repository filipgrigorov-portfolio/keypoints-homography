import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os

np.set_printoptions(suppress=True)

DEBUG = False

def argminwhere(mat):
    rows, cols = mat.shape
    min_vals = []
    min_coords = []
    for col in range(cols):
        min_row = np.argmin(mat[:, col])
        min_coords.append([ min_row, col ])
        min_vals.append(mat[int(min_row)][int(col)])
    
    min_val_abs = [100000, 0]
    for idx in range(len(min_vals)):
        min_val = min_vals[idx]
        if min_val < min_val_abs[0]:
            min_val_abs = [ min_val, min_coords[idx] ]
            
    return np.array([ min_val_abs[1] ])

def keypoints_detection_factory(kpnts_type):
    index_params = dict(algorithm = 0, trees = 5) 
    search_params = dict()
    return cv2.SIFT_create(), cv2.FlannBasedMatcher(index_params, search_params)

class Homography:
    def __init__(self, config):
        self.kpnts_det, self.kpnts_matcher = keypoints_detection_factory(config['keypoints'])
        self.dist_thresh = config['distance_threshold']
        self.prob_outliers = config['probability_of_outliers']
        self.num_pnts_per_sample = config['number_of_keypoints_per_sample']
        self.prob_of_only_inliers = config['probability_of_only_inliers']
        self.N = np.int(np.ceil(np.log(1.0 - self.prob_of_only_inliers) / np.log(1.0 - \
            (1.0 - self.prob_outliers)**self.num_pnts_per_sample)))
        self.eps = config['distance_eps']

        self.render = config['render']
    
    def compute_correspondances_with_sift(self, fr0, fr1):
        kpnts0, descs0 = self.kpnts_det.detectAndCompute(fr0, None)
        kpnts1, descs1 = self.kpnts_det.detectAndCompute(fr1, None)
        
        matches= self.kpnts_matcher.knnMatch(descs0, descs1, k=2) 
            
        good_points=[] 
        for m, n in matches: 
            if(m.distance < self.dist_thresh * n.distance): 
                good_points.append(m)

        if self.render:
            result = cv2.drawMatches(
                fr0, kpnts0, fr1, kpnts1, good_points, 
                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            plt.figure(figsize=(25, 25))
            plt.imshow(result)

        correspondances = []
        for idx in range(len(good_points)):
            match = good_points[idx]
            pnt1 = kpnts0[match.queryIdx]
            pnt2 = kpnts1[match.trainIdx]
            x1, y1 = pnt1.pt
            x2, y2 = pnt2.pt
            correspondances.append([x1, y1, x2, y2])
            
            if idx % 20 == 0:
                print(f'{idx} => {x1} x {y1} - {x2} x {y2}')
        print('\n')
            
        correspondances = np.array(correspondances)
        if DEBUG:
            print(f'\nStacked correspondances:\n{correspondances.shape}\n')
            print(f'\ncorrespondances[0]:\n{correspondances[0]}')
        return correspondances

    def compute_homography_with_ransac(self, correspondances):
        homographies = []
        ninliers = np.zeros(self.N)
        for n_idx in range(self.N):
            random_pairs = correspondances[np.random.choice(correspondances.shape[0], size=self.num_pnts_per_sample)]
            H = self.compute_homography(random_pairs).reshape(3, 3)
            homographies.append(H)
            
            for idx in range(random_pairs.shape[0]):
                pnts0 = np.hstack([random_pairs[idx, :2], 1]).reshape(3, 1)
                pnts1 = np.hstack([random_pairs[idx, 2:], 1]).reshape(3, 1)
                
                pred_pnts1 = H.dot(pnts0)
                pred_pnts1 /= pred_pnts1[-1]
                
                if DEBUG:
                    print('pnts0: ', pnts0.flatten())
                    print('pnts1: ', pnts1.flatten())
                    print('pred pnts1: ', pred_pnts1.flatten())
                    print('\n')
            
                pnt_distance = np.square(pred_pnts1 - pnts1).sum()
                if DEBUG:
                    print(pnt_distance)
                
                if pnt_distance <= self.eps:
                    ninliers[n_idx] += 1
                    
            print(f'\rInliers counts: {ninliers}', end='')
                
        H = homographies[np.argmax(ninliers)]
        if DEBUG:
            print(f'\nHf: {H}')

        return H

    def compute_homography(self, pairs):
        fr_0 = pairs[0][:2]
        fr_1 = pairs[0][2:]
        x0 = fr_0[0]; y0 = fr_0[1]; x1 = fr_1[0]; y1 = fr_1[1]
        A = self.__A_i(x0, y0, x1, y1)
        
        for idx in range(1, pairs.shape[0]):
            fr_0 = pairs[idx][:2]
            fr_1 = pairs[idx][2:]
            x0 = fr_0[0]; y0 = fr_0[1]; x1 = fr_1[0]; y1 = fr_1[1]
            A = np.vstack([A, self.__A_i(x0, y0, x1, y1)])
            
        U, singular_values, Vt = np.linalg.svd(A, full_matrices=True)
        V = Vt.T
        
        if DEBUG:
            print(f'{U.shape}')
            print(f'{singular_values.shape}')
            print(f'{V.shape}')
        
        # Note: Not necessary to compute the minimum of the matrix, as it is sorted in 
        # decreasing order, from max to min => last column is the smallest eigevector
        #min_idx = argminwhere(v)
        #min_coord  = np.argwhere(np.abs(V) == np.abs(V).min())
        return V[:, -1] #V[:, min_coord[0][1]]

    def estimate_camera_position(self, H):
        r1 = H[:, 0]
        r2 = H[:, 1]
        t = H[:, 2]
        r3 = np.cross(r1, r2)
        R = np.array([r1, r2, r3])
        return R, t

    def __A_i(self, x0, y0, x1, y1):
        return np.array([ 
            [-x0, -y0, -1.0, 0.0, 0.0, 0.0, x0 * x1, y0 * x1, x1], 
            [0.0, 0.0, 0.0, -x0, -y0, -1.0, x0 * y1, y0 * y1, y1]])
