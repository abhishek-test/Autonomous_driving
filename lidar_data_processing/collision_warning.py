
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
        img_h, img_w, _ = 375, 1242, 3 #image.shape
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
    velo_points_road = scan_data[:, 0:3] 

    # delete negative liDAR points
    velo_points = np.delete(velo_points, np.where(velo_points[3, :] < 0), axis=1)
    velo_points_road = np.delete(velo_points_road, np.where(velo_points_road[3, :] < 0), axis=1)

    # use ransac to remove ground plane
    if remove_plane:
            ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(),
                        residual_threshold=0.1, max_trials=5000)

            X = velo_points[:, :2]
            y = velo_points[:, -1]
            ransac.fit(X, y)

            # remove outlier points
            mask = ransac.inlier_mask_
            #velo_points = velo_points[~mask]
            velo_points_road = velo_points_road[mask]


    # homogeneous LiDAR points
    velo_points = np.insert(velo_points, 3, 1, axis=1).T
    velo_points_road = np.insert(velo_points_road, 3, 1, axis=1).T 

    return velo_points, velo_points_road


def project_velo2cam(lidar_bin, image, T_mat, remove_plane=False):
    ''' Projects LiDAR point cloud onto the image coordinate frame '''

    # get homogeneous LiDAR points from binn file
    velo_points, velo_points_road = bin2h_velo(lidar_bin, remove_plane)

    # get camera (u, v, z) coordinates
    velo_camera = velo2camera(velo_points, T_mat, image, remove_outliers=True)
    velo_camera_road = velo2camera(velo_points_road, T_mat, image, remove_outliers=True)  

    return velo_camera, velo_camera_road


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
    road_mask = np.zeros((left_image.shape[0], left_image.shape[1]), dtype=np.uint8)    

    (u, v, z), (ur, vr, zr) = project_velo2cam(lidar_bin, left_image, T_mat, remove_plane=True)

    Points = list(zip(u.astype(np.int32), v.astype(np.int32)))    
    col_idx = abs(np.round(64 * 5 / z))
    col_idx = abs(64 - col_idx)

    for i, point in enumerate(Points):
        #cv2.circle(depth_img, point, 5, col_idx[i], -1)
        cv2.circle(depth_img, point, 5, z[i], -1)

    Points_road = list(zip(ur.astype(np.int32), vr.astype(np.int32)))    
    for pointr in Points_road:
        cv2.circle(road_mask, pointr, 5, 255 , -1)

    return depth_img, road_mask


def drawFreeSpace(road_mask, left_image):
    str_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3), (-1,-1))
    copy_image = np.zeros(left_image.shape, left_image.dtype)

    _, binary_img = cv2.threshold(road_mask, 5, 255, cv2.THRESH_BINARY)
    filtered = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, str_kernel, None, (-1, -1))

    contours, hierarchy = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # draw in blue the contours that were founded
        c = max(contours, key = cv2.contourArea)            
        cv2.fillPoly(copy_image, [c], (0,255,0))            

    left_image = cv2.addWeighted(left_image, 1.0, copy_image, 0.2, 1.0)

    return left_image


def drawOnImage(detections, frame, depth_img):

    labels, cord = detections.xyxyn[0][:, -1].to('cpu').numpy(), detections.xyxyn[0][:, :-1].to('cpu').numpy()
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    list_tuple_points = []

    for i in range(n):
        row = cord[i]
        x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
        
        label = f"{int(row[4]*100)}" #conf
        className = names[int(labels[i])]

        c_x = (int)(x1 + ((x2-x1)/2))
        c_y = (int)(y1 + ((y2-y1)/2))

        if( (c_x < 0) or (c_x > frame.shape[1]-1) or (c_y < 0) or (c_y > frame.shape[0]-1) ):
            depth_val = 0
        else: 
            depth_val = depth_img[c_y, c_x]

        txt = className + ": " + str(depth_val) + 'm' #label
        (label_width,label_height), baseline = cv2.getTextSize(txt , cv2.FONT_HERSHEY_COMPLEX, 0.3, 1)
        top_left = tuple(map(int,[int(x1),int(y1)-(label_height+baseline)]))
        top_right = tuple(map(int,[int(x1)+label_width,int(y1)]))
        org = tuple(map(int,[int(x1),int(y1)-baseline]))

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 1)
        cv2.rectangle(frame, top_left, top_right, (255,0,0), -1)

        cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_COMPLEX, 0.3, (255,255,255), 1)
        cv2.circle(frame, (c_x, c_y), 3, (0,255,255), -1)

        pt_tuple = (c_x, c_y)
        list_tuple_points.append(pt_tuple)

    return frame, list_tuple_points


def calcTTC(frame, list_pts):

    half_width = int(frame.shape[1]/2)
    height = frame.shape[0]
    minDist = 1000.0
    minDistIdx = -1

    for i, point in enumerate(list_pts):
        delX = abs(point[0] - half_width)
        if delX < minDist:
            minDist = delX
            minDistIdx = i


    if minDistIdx > -1:
        cv2.line(frame, list_pts[minDistIdx], (half_width, height-1), (0,255,255) )

    return frame


if __name__ == "__main__":
        
    DATA_DIR = 'C:\\Abhishek_Data\\My_Data\\Datasets\\KITTI\\2011_10_03\\'

    # get image and lidar data
    DATA_PATH = DATA_DIR + '2011_10_03_drive_0047_sync'        
    left_image_paths = sorted(glob(os.path.join(DATA_PATH, 'image_02/data/*.png')))
    bin_paths = sorted(glob(os.path.join(DATA_PATH, 'velodyne_points/data/*.bin')))
    T_mat = getTransformMatrix(DATA_DIR)

    # object detection
    names = ['person','bicycle','car','motorcycle','airplane','bus','train','truck', \
             'boat','traffic light','fire hydrant','stop sign','parking meter','bench', \
             'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe', \
             'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
             'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard', \
             'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana', \
             'apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake', \
             'chair','couch','potted plant','bed','dining table','toilet','TV','laptop','mouse', \
             'remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator', \
             'book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
    
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.conf = 0.25
    model.iou = 0.25

    
    for index in range(len(left_image_paths)):

        lidar_bin = bin_paths[index]
        left_image = cv2.imread(left_image_paths[index])     
        rgb_img = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)

        # TODO : add thread 1
        depth_image, road_mask = getDepthImg(lidar_bin, left_image, T_mat)
        left_image = drawFreeSpace(road_mask, left_image)

        # TODO : add thread 2
        detections = model(rgb_img)
        frame, list_pts = drawOnImage(detections, left_image, depth_image)

        frame = calcTTC(frame, list_pts)

        '''
        # output formatting
        depth_image = 3*depth_image
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
        outframe[0:376, 0:1241, :] = frame
        outframe[376:720, 0:1241, :] = depth_image[32:376, 0:1241, :] 
        out.write(outframe)
        '''

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)

        if(key == ord('q')):
            break

        if(key == ord('p')):
            cv2.waitKey(0)