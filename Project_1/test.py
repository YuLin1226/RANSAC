import scale_registration as sr
import random
import numpy as np
import math
import matplotlib.pyplot as plt



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
        pts_np_2 = sr._rot(30).dot(pts_np_1)*ratio + uncertain_np

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

        ratio_ransac = sr._ransac_find_scale(pts_set_1=pts_np_2.T, pts_set_2=pts_np_1.T, sigma=0.05, max_iter=100)

        print("\nNumber %i Iteration: \n- answer: %f\n- solution: %f"%(j+1, ratio_ransac, ratio))
        if abs(ratio - ratio_ransac) < 0.05*ratio:
            times += 1
    
    print("Success Percentage: %f " %(times/n*100))
        # plt.scatter(pts_np_1[0,:], pts_np_1[1,:], c='red')
        # plt.scatter(pts_np_2[0,:], pts_np_2[1,:], c='blue')
        # plt.show()