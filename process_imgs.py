#!/usr/bin/env python

import sys
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

if IN_COLAB:
    print("**** in colab ****")
    if "/content/tracking_dataset_creator" not in sys.path:
        sys.path.insert(0, "/content/tracking_dataset_creator")

    if "/content/tracking_dataset_creator/CSENDistance" not in sys.path:
        sys.path.insert(0, "/content/tracking_dataset_creator/CSENDistance")

    if "/content/tracking_dataset_creator/GCNDepth" not in sys.path:
      sys.path.insert(0, "/content/tracking_dataset_creator/GCNDepth")
    
    print(sys.path)

import os
import numpy as np
from numpy import genfromtxt
import argparse
import yaml
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares, differential_evolution
from tqdm import tqdm
import pandas as pd

from camera_kinematics import CameraKinematics
from range_estimator import RangeEstimator
import utils as utl

import contextily as cx

        
class Position3DVisualizer:

    def __init__(self, path, gt_path, plot_type='ERR', csen_pkg_path='', gcn_pkg_path=''):

        self.path = path

        img_files = utl.get_dataset_imgs(path)
        self._img_files = img_files
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
        self._img_files = [self._img_files[idx] for idx in sub_idxs]
        self._gt_boxes = self._gt_boxes[sub_idxs]

        self.kin = CameraKinematics(66, vis=False)
        self._re = RangeEstimator([self.img_shape[1],self.img_shape[0]], method='direct_fcn')
        self._re_level = RangeEstimator([self.img_shape[1],self.img_shape[0]], method='proportionality')
        self._re_csen = RangeEstimator([self.img_shape[1],self.img_shape[0]], method='direct_csen', csen_pkg_path=csen_pkg_path)
        self._re_gcn = RangeEstimator([self.img_shape[1],self.img_shape[0]], method='direct_gcn', gcn_pkg_path=gcn_pkg_path)



        plt.rcParams["figure.figsize"] = (7,5)
        fig2d, self.ax_2d = plt.subplots()


    def plot_range_errs(self):
        self.ax_2d.plot(range(self._gt_poses_level.shape[0]), np.linalg.norm(self._gt_poses_level[:,0:2]-self._gt_poses_with_alt[:,0:2], axis=1), color=(1,0.64,0), linewidth=2, label='Level assumption')
        self.ax_2d.plot(range(self._gt_poses_with_range.shape[0]), np.linalg.norm(self._gt_poses_with_range[:,0:2]-self._gt_poses_with_alt[:,0:2], axis=1), linewidth=2, label='Range estimation FCN')
        self.ax_2d.plot(range(self._gt_poses_with_range_csen.shape[0]), np.linalg.norm(self._gt_poses_with_range_csen[:,0:2]-self._gt_poses_with_alt[:,0:2], axis=1), linewidth=2, label='Range estimation CSEN')
        self.ax_2d.plot(range(self._gt_poses_with_range_gcn.shape[0]), np.linalg.norm(self._gt_poses_with_range_gcn[:,0:2]-self._gt_poses_with_alt[:,0:2], axis=1), linewidth=2, label='Range estimation GCN')

        self.ax_2d_1.plot(range(self._gt_poses_level.shape[0]), np.abs(self._gt_poses_level[:,2]-self._gt_poses_with_alt[:,2]), color=(1,0.64,0), linewidth=2, label='Level assumption')
        self.ax_2d_1.plot(range(self._gt_poses_with_range.shape[0]), np.abs(self._gt_poses_with_range[:,2]-self._gt_poses_with_alt[:,2]), linewidth=2, label='Range estimation FCN')
        self.ax_2d_1.plot(range(self._gt_poses_with_range_csen.shape[0]), np.abs(self._gt_poses_with_range_csen[:,2]-self._gt_poses_with_alt[:,2]), linewidth=2, label='Range estimation CSEN')
        self.ax_2d_1.plot(range(self._gt_poses_with_range_gcn.shape[0]), np.abs(self._gt_poses_with_range_gcn[:,2]-self._gt_poses_with_alt[:,2]), linewidth=2, label='Range estimation GCN')

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

        # plt.show()
        plt.savefig('range_errs.pdf', format='PDF')


    def to3D(self, index):

        self.new_poses = self._gt_poses.copy()

        img_copy1 = cv.imread( self._img_files[index] )
        img_copy2 = img_copy1.copy()
        img_copy3 = img_copy1.copy()
        img_copy4 = img_copy1.copy()
        img_copy5 = img_copy1.copy()

        box = self._gt_boxes[index].copy()
        direction = self._gt_poses_with_alt[index]-self._drone_poses[index]
        direction = direction/np.linalg.norm(direction)
        self._re.findPos(self._img_files[index], box, direction, self._drone_poses[index,2]) 
        self._re_level.findPos(self._img_files[index], box, direction, self._drone_poses[index,2]) 
        self._re_csen.findPos(self._img_files[index], box, direction, self._drone_poses[index,2]) 
        _, depth = self._re_gcn.findPos(self._img_files[index], box, direction, self._drone_poses[index,2], return_depth_img=True)

        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[4])
        y2 = int(box[5])
        cv.rectangle(img_copy1, (x1, y1), (x2, y2), (0,0,0), 2)
        cv.rectangle(img_copy2, (x1, y1), (x2, y2), (255,0,0), 2)
        cv.rectangle(img_copy3, (x1, y1), (x2, y2), (0,255,0), 2)
        cv.rectangle(img_copy4, (x1, y1), (x2, y2), (0,0,255), 2)
        cv.rectangle(depth, (x1, y1), (x2, y2), (0,255,255), 2)

        data_name = 'example_imgs/' + self.path.split('/')[-2] + '_' + str(index)
        cv.imwrite(data_name+'_gt.jpg', img_copy1)
        cv.imwrite(data_name+'_level.jpg', img_copy2)
        cv.imwrite(data_name+'_fcn.jpg', img_copy3)
        cv.imwrite(data_name+'_csen.jpg', img_copy4)
        cv.imwrite(data_name+'_gcn.jpg', depth)

parser = argparse.ArgumentParser()
parser.add_argument("folder_path", help="Path to data folder")
parser.add_argument("gt_path", help="Path to target groundtruth positions")
parser.add_argument("csen_path", help="Path to CSENDistance package")
parser.add_argument("gcn_path", help="Path to GCNDepth package")

args = parser.parse_args()

opt = Position3DVisualizer(args.folder_path, args.gt_path, plot_type='ERR', \
                           csen_pkg_path=args.csen_path , gcn_pkg_path=args.gcn_path)
opt.to3D(10)
