import random
import numpy as np
import math
import matplotlib.pyplot as plt



def _cal_distance(x1, y1, x2, y2):

    return ((x1-x2)**2 + (y1-y2)**2)**0.5


def _ransac_find_scale(pts_set_1, pts_set_2, sigma, max_iter=1000):

    length, _ = np.shape(pts_set_1)
    best_ratio = 0
    total_inlier, pre_total_inlier = 0, 0

    for i in range(max_iter):
        
        index = random.sample(range(length), 2)
        x11 = pts_set_1[index[0], 0]
        y11 = pts_set_1[index[0], 1]
        x12 = pts_set_1[index[1], 0]
        y12 = pts_set_1[index[1], 1]
        
        x21 = pts_set_2[index[0], 0]
        y21 = pts_set_2[index[0], 1]
        x22 = pts_set_2[index[1], 0]
        y22 = pts_set_2[index[1], 1]

        dist_1 = _cal_distance(x11, y11, x12, y12)
        dist_2 = _cal_distance(x21, y21, x22, y22)

        mean_1 = [(x11+x12)/2, (y11+y12)/2]
        mean_2 = [(x21+x22)/2, (y21+y22)/2]

        ratio = dist_1 / dist_2
        
        for j in range(length):
            x1j = pts_set_1[j,0]
            y1j = pts_set_1[j,1]
            x2j = pts_set_2[j,0]
            y2j = pts_set_2[j,1]

            dist_1_j = _cal_distance(mean_1[0], mean_1[1], x1j, y1j)
            dist_2_j = _cal_distance(mean_2[0], mean_2[1], x2j, y2j)

            ratio_j = dist_1_j / dist_2_j

            if abs(ratio - ratio_j) < sigma:

                total_inlier += 1

        if total_inlier > pre_total_inlier:
            
            pre_total_inlier = total_inlier
            best_ratio = ratio
        
        total_inlier = 0
    return best_ratio


def rot(theta):

    x = theta / 180 * math.pi
    T = np.array([
        [math.cos(x), -math.sin(x)],
        [math.sin(x),  math.cos(x)]
    ])

    return T

if __name__ == "__main__":
    
    ratio = 3
    n = 1000
    times = 0
    for j in range(n):
        pts = []
        uncertain = []
        for i in range(30):
            pts.append([
                random.randint(1, 100),
                random.randint(1, 100)
            ]) 

            uncertain.append([
                random.uniform(1.1, 5.4),
                random.uniform(1.1, 5.4)
            ])
        
        pts_np_1 = np.array(pts).T
        uncertain_np = np.array(uncertain).T
        pts_np_2 = rot(0).dot(pts_np_1)*ratio + uncertain_np

        noise_1, noise_2 = [], []
        for i in range(20):
            noise_1.append([
                random.randint(1, 300),
                random.randint(1, 300)
            ])
            noise_2.append([
                random.randint(1, 500),
                random.randint(1, 500)
            ])

        pts_np_1 = np.hstack((pts_np_1, np.array(noise_1).T))    
        pts_np_2 = np.hstack((pts_np_2, np.array(noise_2).T))

        ratio_ransac = _ransac_find_scale(pts_set_1=pts_np_2.T, pts_set_2=pts_np_1.T, sigma=0.05, max_iter=100)

        print("\nNumber %i Iteration: \n- answer: %f\n- solution: %f"%(j+1, ratio_ransac, ratio))
        if abs(ratio - ratio_ransac) < 0.05*ratio:
            times += 1
    
    print("Success Percentage: %f " %(times/n*100))
        # plt.scatter(pts_np_1[0,:], pts_np_1[1,:], c='red')
        # plt.scatter(pts_np_2[0,:], pts_np_2[1,:], c='blue')
        # plt.show()