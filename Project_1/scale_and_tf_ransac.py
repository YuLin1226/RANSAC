# -*- coding:utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import random


def scale(scale=2.0):
    T = np.array([
        [scale ,     0],
        [    0 , scale]
    ])
    return T

def rot(theta=90):
    """
    theta: unit: degree
    """
    yaw = theta/180*math.pi
    T = np.array([
        [math.cos(yaw), math.sin(yaw)],
        [-math.sin(yaw), math.cos(yaw)]
    ])
    return T

def trans(pts, dx=0, dy=0):
    row, col = np.shape(pts)
    x = np.ones((1,col))*dx
    y = np.ones((1,col))*dy
    ds = np.vstack((x,y))
    return ds + pts


def _cal_distance(x1, y1, x2, y2):
        return ((x1-x2)**2 + (y1-y2)**2)**0.5

def _ransac_find_rotation_translation(pts_set_1, pts_set_2, sigma=1, max_iter=1000):
        """
        - pts_set_1: Nx2 ndarray. \n
        - pts_set_2: Nx2 ndarray. \n
        - sigma: error threshold. \n
        - max_iter: Maximum times of the iteration
        """
        length, _ = np.shape(pts_set_1)
        best_relation = None
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

            v1 = [x11- x12, y11 -y12]
            u1 = [x21- x22, y21 -y22]

            dtheta =  (math.atan2(v1[1], v1[0]) - math.atan2(u1[1], u1[0]))/math.pi*180
            ratio = _cal_distance(u1[0], u1[1], 0, 0)/_cal_distance(v1[0], v1[1], 0, 0)
            

            ds = np.array([[x21],[y21]]) - rot(dtheta).dot(scale(ratio).dot(np.array([[x11], [y11]])))
            

            for j in range(length):
                x1j = pts_set_1[j,0]
                y1j = pts_set_1[j,1]
                x2j = pts_set_2[j,0]
                y2j = pts_set_2[j,1]

                pt1j = np.array([
                                    [x1j],
                                    [y1j]
                                ])
                pt2j = np.array([
                                    [x2j],
                                    [y2j]
                                ])
                ptj_error = pt2j - scale(ratio).dot(rot(dtheta).dot(pt1j)) - ds

                if _cal_distance(x1=0, y1=0, x2=ptj_error[0], y2=ptj_error[1]) < sigma:

                    total_inlier += 1

            if total_inlier > pre_total_inlier:
                
                pre_total_inlier = total_inlier
                dx, dy = ds[0], ds[1]
                best_relation = [ratio, dx, dy, dtheta]
            
            total_inlier = 0
        return best_relation


num = 50
pts = []
for i in range(num):
    pts.append([    
                random.uniform(1.0 , 50.0),
                random.uniform(1.0 , 50.0)
                ])
pts = np.array(pts).T # 2xN

pts_2 = scale(scale=3.0).dot(pts)
pts_2 = rot(theta=30).dot(pts_2)
pts_3 = trans(pts=pts_2, dx=70, dy=-70) # 2xN




for i in range(10):
    best_relation = _ransac_find_rotation_translation(pts_set_1=pts.T, pts_set_2=pts_3.T)
    print("\nRatio:  %f\nDisplacement:  %f , %f\nOrientation: %f"%(best_relation[0], best_relation[1], best_relation[2], best_relation[3]))



