#!/usr/bin/env python

import os
import numpy as np
from numpy import genfromtxt
import argparse
import yaml
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares, differential_evolution

from camera_kinematics import CameraKinematics
from range_estimator import RangeEstimator
import utils as utl

import contextily as cx

import sys
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

if IN_COLAB:
    # print("**** in colab ****")
    if "/content/tracking_dataset_creator" not in sys.path:
        # print("**** path not set ****")
        sys.path.insert(0, "/content/tracking_dataset_creator")
        # print(sys.path)
        
class Position3DVisualizer:

    def __init__(self, path, gt_path, plot_type='ERR'):

        self.path = path

        img_files = utl.get_dataset_imgs(path)
        num_imgs = len(img_files)
        self.img_shape = cv.imread(img_files[0]).shape
        self._gt_boxes = genfromtxt(path+"/groundtruth.txt", delimiter=',')
        self._f = (0.5 * self.img_shape[1] * (1.0 / np.tan((66.0/2.0)*np.pi/180)));

        gt_data = genfromtxt(gt_path, delimiter=',')

        ref_loc = gt_data[0,1:4]
        gt_alts = gt_data[:,3].copy()
        gt_data[:,3] = 0

        stats_data = genfromtxt(path+"/camera_states.txt", delimiter=',')

        drone_poses = utl.gps_to_ned(ref_loc, stats_data[:,1:4])
        gt_poses = utl.gps_to_ned(ref_loc, gt_data[:,1:4])
        self._gt_poses, sub_idxs = utl.interpolate(stats_data[:,0], gt_data[:,0], \
                                                gt_poses, synced=True)
        self._gt_alts, _ = utl.interpolate(stats_data[:,0], gt_data[:,0], \
                                                gt_alts.reshape((gt_alts.shape[0],1)), synced=True)
        self._gt_poses_with_alt = self._gt_poses.copy()
        self._gt_poses_with_alt[:,2] = self._gt_alts[:,0] - self._gt_alts[0,0]

        sub_idxs = np.array(sub_idxs)
        sub_idxs = sub_idxs[sub_idxs<num_imgs]
        self._gt_poses = self._gt_poses[0:sub_idxs.shape[0]]
        self._gt_poses_with_alt = self._gt_poses_with_alt[0:sub_idxs.shape[0]]
        self._drone_poses = drone_poses[sub_idxs]
        self._gt_boxes = self._gt_boxes[sub_idxs]

        self.kin = CameraKinematics(66, vis=False)
        self._re = RangeEstimator([self.img_shape[1],self.img_shape[0]], method='direct')
        self._re_level = RangeEstimator([self.img_shape[1],self.img_shape[0]], method='proportionality')

        assert plot_type in ['3D', '2D', 'ERR', 'NONE']
        self.plot_type = plot_type

        if plot_type == '3D':
            fig_3d=plt.figure(0)
            self.ax_3d=plt.axes(projection ='3d')
            self.ax_3d.set_title('Trajectories Plot')

        elif plot_type == '2D':
            plt.rcParams["figure.figsize"] = (7,5)
            fig2d, self.ax_2d = plt.subplots()

        elif plot_type == 'ERR':
            plt.rcParams["figure.figsize"] = (7,5)
            fig2d, self.ax_2d = plt.subplots()

            plt.rcParams["figure.figsize"] = (7,5)
            fig2d_1, self.ax_2d_1 = plt.subplots()

        elif plot_type == 'ERR':
            pass


    def plot_trajs(self):

        self.ax_3d.cla()

        self.ax_3d.plot(self._gt_poses[:,0],self._gt_poses[:,1],self._gt_poses[:,2], color=(1,0.64,0), label='Level Target Position')
        self.ax_3d.plot(self._gt_poses_with_alt[:,0],self._gt_poses_with_alt[:,1],self._gt_poses_with_alt[:,2], color=(0,100.0/255,100.0/255), label='Real Target Position')
        self.ax_3d.plot(self._gt_poses_with_range[:,0],self._gt_poses_with_range[:,1],self._gt_poses_with_range[:,2], color=(0,1,0), label='Target Position Calculated by Range')
        # self.ax_3d.plot(self._drone_poses[:,0],self._drone_poses[:,1],self._drone_poses[:,2], color='black', label='Camera Position')

        self.ax_3d.set_xlabel('x')
        self.ax_3d.set_ylabel('y')
        self.ax_3d.set_zlabel('z')

        self.ax_3d.legend()

        utl.set_axes_equal(self.ax_3d)

        # self.ax_3d.set_xlim(-20,20)
        # self.ax_3d.set_ylim(-45,10)
        self.ax_3d.set_zlim(-5,5)

        plt.show()

    def plot_trajs_2d(self):

        self.ax_2d.cla()

        gt_hd = self.ax_2d.plot(self._gt_poses[:,0],self._gt_poses[:,1], color=(1,0.64,0), linewidth=2, label='Raw Target Position')
        dr_hd = self.ax_2d.plot(self._drone_poses[:,0],self._drone_poses[:,1], color='black', linewidth=2, label='Camera Position')

        for i in range(self._gt_poses.shape[0]):
            if i%150 == 50:
                utl.add_arrow(gt_hd[0], position=i, size=20)

        self.ax_2d.set_xlabel('x - north (m)')
        self.ax_2d.set_ylabel('y - east (m)')

        self.ax_2d.legend()
        self.ax_2d.grid()

        # utl.set_axes_equal(self.ax_3d)

        # self.ax_3d.set_xlim(-20,20)
        # self.ax_3d.set_ylim(-45,10)
        # self.ax_3d.set_zlim(-15,10)

        plt.tight_layout()
        cx.add_basemap(self.ax_2d, crs=NL_shape.crs, source=cx.providers.OpenStreetMap.Mapnik)
        plt.show()

    def plot_range_errs(self):
        self.ax_2d.plot(range(self._gt_poses_level.shape[0]), np.linalg.norm(self._gt_poses_level[:,0:2]-self._gt_poses_with_alt[:,0:2], axis=1), color=(1,0.64,0), linewidth=2, label='Level assumption')
        self.ax_2d.plot(range(self._gt_poses_with_range.shape[0]), np.linalg.norm(self._gt_poses_with_range[:,0:2]-self._gt_poses_with_alt[:,0:2], axis=1), linewidth=2, label='Range estimation')

        self.ax_2d_1.plot(range(self._gt_poses_level.shape[0]), np.abs(self._gt_poses_level[:,2]-self._gt_poses_with_alt[:,2]), color=(1,0.64,0), linewidth=2, label='Level assumption')
        self.ax_2d_1.plot(range(self._gt_poses_with_range.shape[0]), np.abs(self._gt_poses_with_range[:,2]-self._gt_poses_with_alt[:,2]), linewidth=2, label='Range estimation')

        self.ax_2d.set_xlabel('Frame Counter')
        self.ax_2d.set_ylabel('Horizontal Position Error (m)')

        self.ax_2d_1.set_xlabel('Frame Counter')
        self.ax_2d_1.set_ylabel('Vertical Position Error (m)')

        self.ax_2d.legend()
        self.ax_2d.grid()
        self.ax_2d.set_xlim([0,self._gt_poses_level.shape[0]])

        self.ax_2d_1.legend()
        self.ax_2d_1.grid()
        self.ax_2d_1.set_xlim([0,self._gt_poses_level.shape[0]])

        plt.show()


    def to3D(self, visualize=False):

        self.new_poses = self._gt_poses.copy()

        self._gt_poses_with_range = []
        self._gt_poses_level = []
        for i,box in enumerate(self._gt_boxes):
            direction = self._gt_poses_with_alt[i]-self._drone_poses[i]
            direction = direction/np.linalg.norm(direction)
            self._gt_poses_with_range.append( self._re.findPos(box, direction, self._drone_poses[i,2]) + self._drone_poses[i] )
            self._gt_poses_level.append( self._re_level.findPos(box, direction, self._drone_poses[i,2]) + self._drone_poses[i] )


        self._gt_poses_with_range = np.array(self._gt_poses_with_range)
        self._gt_poses_level = np.array(self._gt_poses_level)

        if visualize:
            # self.plot_trajs_2d()
            # self.plot_trajs()
            self.plot_range_errs()


parser = argparse.ArgumentParser()
parser.add_argument("folder_path", help="Path to data folder")
parser.add_argument("gt_path", help="Path to target groundtruth positions")

args = parser.parse_args()

# opt = Position3DVisualizer(args.folder_path, args.gt_path)
# opt.to3D()
acts = []
ests = []
ests1 = []

for i in range(1,8):
    # del opt

    folder_path = list(args.folder_path)
    gt_path = list(args.gt_path)

    num_idx = args.folder_path.rfind('/')-1
    folder_path[num_idx] = str(i)

    num_idx = args.gt_path.find('.txt')-1
    gt_path[num_idx] = str(i)

    folder_path = ''.join(folder_path)
    gt_path = ''.join(gt_path)
    print(folder_path, gt_path)
    opt = Position3DVisualizer(folder_path, gt_path, plot_type='NONE')
    opt.to3D()

    act = np.linalg.norm(opt._gt_poses_with_alt[:]-opt._drone_poses[:], axis=1)
    est = np.linalg.norm(opt._gt_poses_level[:]-opt._drone_poses[:], axis=1)
    est1 = np.linalg.norm(opt._gt_poses_with_range[:]-opt._drone_poses[:], axis=1)

    acts.append(act)
    ests.append(est)
    ests1.append(est1)

plt.rcParams["figure.figsize"] = (5,5)
fig, ax = plt.subplots()

ax.plot([0,30], [0,30], color='red', linewidth=2)

acts = np.hstack(acts)
ests = np.hstack(ests)
ests1 = np.hstack(ests1)


ax.scatter(acts, ests, color='blue', s=4)
ax.scatter(acts, ests1, color='orange', s=4)

std = np.std(ests-acts)
ax.fill_between(np.arange(0,30), np.arange(0,30)+std, np.arange(0,30)-std, color='red', alpha=0.3)

ax.set_xlabel('Actual Distance (m)')
ax.set_ylabel('Estimated Distance (m)')

ax.set_xlim([0,30])
ax.set_ylim([0,30])

ax.legend()
ax.grid()

plt.show()
