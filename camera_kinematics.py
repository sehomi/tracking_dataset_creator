#!/usr/bin/env python

import time
import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R
from utils import plot_kinematics
import matplotlib.pyplot as plt

class CameraKinematics:

    def __init__(self, hfov, vis=True):

        self._hfov = hfov
        self._init = False

        self._vis=vis
        if vis:
            self._fig_3d=plt.figure(0)
            self._ax_3d=plt.axes(projection ='3d')
            self._ax_3d.set_title('Kinematics Plot')

    def init_params(self, w, h):

        self._cx = w/2
        self._cy = h/2
        self._w = w
        self._h = h
        self._f = (0.5 * w * (1.0 / np.tan((self._hfov/2.0)*np.pi/180)));

    def body_to_inertia(self, body_vec, eul):

        if body_vec is None:
            return None

        ## calculate a DCM and find transpose that takes body to inertial
        DCM_ib = self.make_DCM(eul).T

        ## return vector in inertial coordinates
        return np.matmul(DCM_ib, body_vec)

    def inertia_to_body(self, in_vec, eul):

        ## calculate a "DCM" using euler angles of camera body, to convert vector
        ## from inertial to body coordinates
        DCM_bi = self.make_DCM(eul)

        ## return the vector in body coordinates
        return np.matmul(DCM_bi, in_vec)

    def cam_to_body(self, rect):

        if rect is None:
            return None

        ## converting 2d rectangle to a 3d vector in camera coordinates
        vec = self.to_direction_vector(rect, self._cx, self._cy, self._f)

        ## for MAVIC Mini camera, the body axis can be converted to camera
        ## axis by a 90 deg yaw and a 90 deg roll consecutively. then we transpose
        ## it to get camera to body
        DCM_bc = self.make_DCM([90*np.pi/180, 0, 90*np.pi/180]).T

        return np.matmul(DCM_bc, vec)

    def body_to_cam(self, vec):

        ## for MAVIC Mini camera, the body axis can be converted to camera
        ## axis by a 90 deg yaw and a 90 deg roll consecutively.
        DCM_cb = self.make_DCM([90*np.pi/180, 0, 90*np.pi/180])

        return np.matmul(DCM_cb, vec)

    def to_direction_vector(self, rect, cx, cy, f):

        ## find center point of target
        center = np.array([rect[0]+rect[2]/2, rect[1]+rect[3]/2])

        ## project 2d point from image plane to 3d space using a simple pinhole
        ## camera model
        w = np.array( [ (center[0] - cx) , (center[1] - cy), f] )
        return w/np.linalg.norm(w)

    def from_direction_vector(self, dir, cx, cy, f):

        ## avoid division by zero
        if dir[2] < 0.01:
            dir[2] = 0.01

        ## calculate reprojection of direction vectors to image plane using a
        ## simple pinhole camera model
        X = cx + (dir[0] / dir[2]) * f
        Y = cy + (dir[1] / dir[2]) * f

        return (int(X),int(Y))

    def reproject(self, drone_poses, gt_poses, euls, imgs):
        dirs = gt_poses - drone_poses

        first_img = cv.imread(imgs[0])

        w = first_img.shape[1]
        h = first_img.shape[0]
        self.init_params(w, h)

        for i,img_name in enumerate(imgs):

            dir = dirs[i]/np.linalg.norm(dirs[i])
            imu_meas = euls[i,:]

            # print(img_name, euls[i,:])

            body_dir_est = self.inertia_to_body(dir,imu_meas)
            cam_dir_est = self.body_to_cam(body_dir_est)
            center_est = self.from_direction_vector(cam_dir_est, self._cx, self._cy, self._f)
            rect = (center_est[0]-1,center_est[1]-1,2,2)
            corners = self.get_camera_frame_vecs(imu_meas,self._w,self._h)
            plot_kinematics(imu_meas,dir,self._ax_3d,corners)

            frame = cv.imread(img_name)

            cv.rectangle(frame, rect, (0,255,255), 2)
            cv.imshow("image", frame)
            cv.waitKey(33)

    def get_camera_frame_vecs(self, eul, w, h):

        ## convert image corners from a point in "image coordinates" to a vector
        ## in "camera body coordinates"
        top_left = self.cam_to_body([-1,-1,2,2])
        top_right = self.cam_to_body([w-1,-1,2,2])
        bottom_left = self.cam_to_body([-1,h-1,2,2])
        bottom_right = self.cam_to_body([w-1,h-1,2,2])

        ## convert image corners from a vector in "camera body coordinates" to
        ## a vector in "inertial coordinates"
        top_left_inertia_dir = self.body_to_inertia(top_left, eul)
        top_right_inertia_dir = self.body_to_inertia(top_right, eul)
        bottom_left_inertia_dir = self.body_to_inertia(bottom_left, eul)
        bottom_right_inertia_dir = self.body_to_inertia(bottom_right, eul)


        return (top_left_inertia_dir,top_right_inertia_dir,\
                bottom_left_inertia_dir,bottom_right_inertia_dir)

    def make_DCM(self, eul):

        phi = eul[0]
        theta = eul[1]
        psi = eul[2]

        DCM = np.zeros((3,3))
        DCM[0,0] = np.cos(psi)*np.cos(theta)
        DCM[0,1] = np.sin(psi)*np.cos(theta)
        DCM[0,2] = -np.sin(theta)
        DCM[1,0] = np.cos(psi)*np.sin(theta)*np.sin(phi)-np.sin(psi)*np.cos(phi)
        DCM[1,1] = np.sin(psi)*np.sin(theta)*np.sin(phi)+np.cos(psi)*np.cos(phi)
        DCM[1,2] = np.cos(theta)*np.sin(phi)
        DCM[2,0] = np.cos(psi)*np.sin(theta)*np.cos(phi)+np.sin(psi)*np.sin(phi)
        DCM[2,1] = np.sin(psi)*np.sin(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi)
        DCM[2,2] = np.cos(theta)*np.cos(phi)

        return DCM
