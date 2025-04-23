
import os
import cv2
import torch
import numpy as np
import pandas as pd
from glob import glob
from sklearn import linear_model

def velo2camera(velo_points, T_mat, image=None, remove_outliers=True):
    ''' maps velo points (LiDAR) to camera (u,v,z) space '''
    # convert to (left) camera coordinates
    # P_left @ R_left_rect @ T_cam_velo
    velo_camera =  T_mat @ velo_points

    # delete negative camera points ??
    velo_camera  = np.delete(velo_camera , np.where(velo_camera [2,:] < 0)[0], axis=1) 

    # get camera coordinates u,v,z
    velo_camera[:2] /= velo_camera[2, :]

    # remove outliers (points outside of the image frame)
    if remove_outliers:
        u, v, z = velo_camera
        img_h, img_w, _ = 375,1242, 3 #image.shape
        u_out = np.logical_or(u < 0, u >= img_w)
        v_out = np.logical_or(v < 0, v >= img_h)
        outlier = np.logical_or(u_out, v_out)
        velo_camera = np.delete(velo_camera, np.where(outlier), axis=1)    

    return velo_camera

def bin2h_velo(lidar_bin, remove_plane=True):
    ''' Reads LiDAR bin file and returns homogeneous (x,y,z,1) LiDAR points'''
    # read in LiDAR data
    scan_data = np.fromfile(lidar_bin, dtype=np.float32).reshape((-1,4))

    # get x,y,z LiDAR points (x, y, z) --> (front, left, up)
    velo_points = scan_data[:, 0:3] 

    # delete negative liDAR points
    velo_points = np.delete(velo_points, np.where(velo_points[3, :] < 0), axis=1)

    # use ransac to remove ground plane
    if remove_plane:
            ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(),
                        residual_threshold=0.1, max_trials=50)

            X = velo_points[:, :2]
            y = velo_points[:, -1]
            ransac.fit(X, y)

            # remove outlier points
            mask = ransac.inlier_mask_
            velo_points = velo_points[mask] #velo_points[~mask]

    # homogeneous LiDAR points
    velo_points = np.insert(velo_points, 3, 1, axis=1).T 

    return velo_points

def project_velo2cam(lidar_bin, image, T_mat, remove_plane=False):
    ''' Projects LiDAR point cloud onto the image coordinate frame '''

    # get homogeneous LiDAR points from binn file
    velo_points = bin2h_velo(lidar_bin, remove_plane=True)

    # get camera (u, v, z) coordinates
    velo_camera = velo2camera(velo_points, T_mat, image, remove_outliers=True)

    return velo_camera

def getTransformMatrix(DATA_DIR):

    # Get Cam Calib Data
    with open(DATA_DIR + 'calib_cam_to_cam.txt','r') as f:
        calib = f.readlines()

    # get projection matrices
    P_left = np.array([float(x) for x in calib[25].strip().split(' ')[1:]]).reshape((3,4))

    # get rectified rotation matrices
    R_left_rect = np.array([float(x) for x in calib[24].strip().split(' ')[1:]]).reshape((3, 3,))
    R_left_rect = np.insert(R_left_rect, 3, values=[0,0,0], axis=0)
    R_left_rect = np.insert(R_left_rect, 3, values=[0,0,0,1], axis=1)

    with open(DATA_DIR + 'calib_velo_to_cam.txt', 'r') as f:
        calib = f.readlines()

    R_cam_velo = np.array([float(x) for x in calib[1].strip().split(' ')[1:]]).reshape((3, 3))
    t_cam_velo = np.array([float(x) for x in calib[2].strip().split(' ')[1:]])[:, None]
    T_cam_velo = np.vstack((np.hstack((R_cam_velo, t_cam_velo)), np.array([0, 0, 0, 1])))

    # matrix to transform from velo (LiDAR) to left color camera
    T_mat = P_left @ R_left_rect @ T_cam_velo

    return T_mat

def getDepthImg(lidar_bin, left_image, T_mat):

    depth_img = np.zeros((left_image.shape[0], left_image.shape[1]), dtype=np.uint8)      
    (u, v, z) = project_velo2cam(lidar_bin, left_image, T_mat, remove_plane=False)

    for i in range(len(u)):
        c_x = (int)(u[i])
        c_y = int(v[i])
        c_z = int(z[i])

        cv2.circle(depth_img, (c_x, c_y), 3, (c_z), -1)

    return 3*depth_img


if __name__ == "__main__":
        
    DATA_DIR = 'C:\\Abhishek_Data\\My_Data\\Datasets\\KITTI\\2011_10_03\\'

    # get image and lidar data
    DATA_PATH = DATA_DIR + '2011_10_03_drive_0047_sync'        
    left_image_paths = sorted(glob(os.path.join(DATA_PATH, 'image_02/data/*.png')))
    bin_paths = sorted(glob(os.path.join(DATA_PATH, 'velodyne_points/data/*.bin')))
    T_mat = getTransformMatrix(DATA_DIR)

    str_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3), (-1,-1))
     
    for index in range(len(left_image_paths)):

        lidar_bin = bin_paths[index]
        left_image = cv2.imread(left_image_paths[index])  
        copy_image = np.zeros(left_image.shape, left_image.dtype)  
        rgb_img = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)

        depth_image = getDepthImg(lidar_bin, left_image, T_mat)
        _, binary_img = cv2.threshold(depth_image, 5, 255, cv2.THRESH_BINARY)
        filtered = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, str_kernel, None, (-1, -1))

        contours, hierarchy = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) != 0:
            # draw in blue the contours that were founded
            c = max(contours, key = cv2.contourArea)            
            cv2.fillPoly(copy_image, [c], (0,255,0))            

        frame = cv2.addWeighted(left_image, 1.0, copy_image, 0.2, 1.0)    
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)

        if(key == ord('q')):
            break

        if(key == ord('p')):
            cv2.waitKey(0)