
import os
import cv2
import torch
import numpy as np
import pandas as pd
from glob import glob
from sklearn import linear_model
from timeit import default_timer as timer

color64 = [
    (0, 0, 0.5625),
    (0, 0, 0.6250),
    (0, 0, 0.6875),
    (0, 0, 0.7500),
    (0, 0, 0.8125),
    (0, 0, 0.8750),
    (0, 0, 0.9375),
    (0, 0, 1),
    (0, 0.0625, 1),
    (0, 0.1250, 1),
    (0, 0.1875, 1),
    (0, 0.2500, 1),
    (0, 0.3125, 1),
    (0, 0.3750, 1),
    (0, 0.4375, 1),
    (0, 0.5000, 1),
    (0, 0.5625, 1),
    (0, 0.6250, 1),
    (0, 0.6875, 1),
    (0, 0.7500, 1),
    (0, 0.8125, 1),
    (0, 0.8750, 1),
    (0, 0.9375, 1),
    (0, 1, 1),
    (0.0625, 1, 0.9375),
    (0.1250, 1, 0.8750),
    (0.1875, 1, 0.8125),
    (0.2500, 1, 0.7500),
    (0.3125, 1, 0.6875),
    (0.3750, 1, 0.6250),
    (0.4375, 1, 0.5625),
    (0.5000, 1, 0.5000),
    (0.5625, 1, 0.4375),
    (0.6250, 1, 0.3750),
    (0.6875, 1, 0.3125),
    (0.7500, 1, 0.2500),
    (0.8125, 1, 0.1875),
    (0.8750, 1, 0.1250),
    (0.9375, 1, 0.0625),
    (1, 1, 0),
    (1, 0.9375, 0),
    (1, 0.8750, 0),
    (1, 0.8125, 0),
    (1, 0.7500, 0),
    (1, 0.6875, 0),
    (1, 0.6250, 0),
    (1, 0.5625, 0),
    (1, 0.5000, 0),
    (1, 0.4375, 0),
    (1, 0.3750, 0),
    (1, 0.3125, 0),
    (1, 0.2500, 0),
    (1, 0.1875, 0),
    (1, 0.1250, 0),
    (1, 0.0625, 0),
    (1, 0, 0),
    (0.9375, 0, 0),
    (0.8750, 0, 0),
    (0.8125, 0, 0),
    (0.7500, 0, 0),
    (0.6875, 0, 0),
    (0.6250, 0, 0),
    (0.5625, 0, 0),
    (0.5000, 0, 0)
]

def getUVDisparity(depth_image, maxDisparity):

    width = depth_image.shape[1]
    height = depth_image.shape[0]

    vDisp = np.zeros( (height, maxDisparity), dtype=np.float32)
    uDisp = np.zeros( (maxDisparity, width), dtype=np.float32)

    for i in range(height):
        vDisp[i, ...] = cv2.calcHist(images=[depth_image[i, ...]], channels=[0], 
                mask=None, histSize=[maxDisparity], ranges=[0, maxDisparity]).flatten() / float(height)    

    for i in range(width):
        uDisp[..., i] = cv2.calcHist(images=[depth_image[..., i]], channels=[0], 
                mask=None, histSize=[maxDisparity], ranges=[0, maxDisparity]).flatten() / float(width)

    vDisp = (vDisp * 255).astype(np.uint8)
    uDisp = (uDisp * 255).astype(np.uint8)

    _, uDisp = cv2.threshold(uDisp, 5, 255, cv2.THRESH_BINARY)
    _, vDisp = cv2.threshold(vDisp, 50, 255, cv2.THRESH_BINARY)

    return uDisp, vDisp

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
        img_h, img_w, _ = image.shape # 375, 1242, 3 
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
                        residual_threshold=0.1, max_trials=5000)

            X = velo_points[:, :2]
            y = velo_points[:, -1]
            ransac.fit(X, y)

            # remove outlier points
            mask = ransac.inlier_mask_
            velo_points = velo_points[~mask]

    # homogeneous LiDAR points
    velo_points = np.insert(velo_points, 3, 1, axis=1).T 

    return velo_points

def project_velo2cam(lidar_bin, image, T_mat, remove_plane=False):
    ''' Projects LiDAR point cloud onto the image coordinate frame '''

    # get homogeneous LiDAR points from binn file
    velo_points = bin2h_velo(lidar_bin, remove_plane)

    # get camera (u, v, z) coordinates
    velo_camera = velo2camera(velo_points, T_mat, image, remove_outliers=True)  

    return velo_camera

def decompose_projection_matrix(P):    
    K, R, T, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    T = T/T[3]

    return K, R, T

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

    K_left, R_left, T_left = decompose_projection_matrix(P_left)

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
    (u, v, z) = project_velo2cam(lidar_bin, left_image, T_mat, remove_plane=False) # 75 ms

    Points = list(zip(u.astype(np.int32), v.astype(np.int32)))    
    col_idx = abs(np.round(64 * 5 / z))
    col_idx = abs(64 - col_idx)

    for i, point in enumerate(Points):
        cv2.circle(depth_img, point, 5, col_idx[i], -1)

    '''
    col_idx = abs(np.round(64 * 5 / z))
    col_idx = abs(64 - col_idx)
    depth_img[v.astype(np.int32), u.astype(np.int32)] = col_idx
    depth_img = (depth_img * 255).astype(np.uint8)

    #******************************
    for i in range(len(u)):  # 180 ms
        c_x = (int)(u[i])
        c_y = int(v[i])
        c_z = int(z[i])

        if(c_y < 180):
            continue

        col_idx = abs(round(64 * 5 / (c_z)))

        if (col_idx >= 64):
            col_idx = 63
            
        if (col_idx <= 0):
            col_idx = 0

        col_idx = abs(64 - col_idx)

        r = np.ceil(255*color64[col_idx][2])
        g = np.ceil(255*color64[col_idx][1])
        b = np.ceil(255*color64[col_idx][0])		
        color = (b,g,r)
        color_gray = 0.299*r + 0.587*g + 0.114*b
        
        cv2.circle(depth_img, (c_x, c_y), 5, col_idx, -1)
        '''
            
    return depth_img


if __name__ == "__main__":
        
    DATA_DIR = 'C:\\Abhishek_Data\\My_Data\\Datasets\\KITTI\\2011_10_03\\'

    # get image and lidar data
    DATA_PATH = DATA_DIR + '2011_10_03_drive_0047_sync'        
    left_image_paths = sorted(glob(os.path.join(DATA_PATH, 'image_02/data/*.png')))
    bin_paths = sorted(glob(os.path.join(DATA_PATH, 'velodyne_points/data/*.bin')))
    T_mat = getTransformMatrix(DATA_DIR)

    fps_patch = np.zeros((30, 150, 3), dtype=np.uint8)

    for index in range(len(left_image_paths)):        

        start = timer()

        lidar_bin = bin_paths[index]
        left_image = cv2.imread(left_image_paths[index])     
        rgb_img = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
              
        depth_image = getDepthImg(lidar_bin, left_image, T_mat) 
        uDisp, vDisp = getUVDisparity(depth_image, maxDisparity=64)   

        end = timer()      
        
        time = 1000.0 * (end-start)

        fps_patch[:,:,:] = 0
        cv2.putText(fps_patch, "Time: " + str(int(time)), (5,20), 
            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,255))
        
        frame = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)

        frame[30:60, 1050:1200, :] = fps_patch   
        
        cv2.imshow("Frame", frame)
        cv2.imshow("U_Disp", uDisp)
        cv2.imshow("V_Disp", vDisp)
        
        key = cv2.waitKey(1)

        if(key == ord('q')):
            break

        if(key == ord('p')):
            cv2.waitKey(0)