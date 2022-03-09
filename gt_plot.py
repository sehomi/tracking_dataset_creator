#!/usr/bin/env python

import os
import numpy as np
from numpy import genfromtxt
import argparse
import yaml
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from camera_kinematics import CameraKinematics
import utils as utl


class GTPlot:

    def __init__(self, args):
        self._args = args

    def process_mavic_with_camera_gt(self):

        gt_data = genfromtxt(self._args.processed_gt_file, delimiter=',')
        gt_data = np.insert(gt_data, 2, 0, axis=1)

        data_dir, _ = os.path.split(self._args.test_config_file)
        log_data = genfromtxt(data_dir+"/log.txt", delimiter=',')

        with open(self._args.test_config_file, 'r') as stream:
            test_config = yaml.safe_load(stream)

        bl_loc = test_config["p_bottom_left_ned"]
        angle = test_config["field_angle"]

        drone_poses = utl.gps_to_ned(bl_loc, log_data[:,2:5])
        gt_poses = utl.interpolate(log_data[:,1], gt_data[:,3], gt_data[:,:3])
        gt_poses = utl.field_to_ned(gt_poses, angle)

        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')
        # ax.plot(gt_data[:,0], gt_data[:,1], 0, linewidth=2)
        ax.plot(gt_poses[:,0], gt_poses[:,1], 0, linewidth=2)
        ax.plot(drone_poses[:,0], drone_poses[:,1], drone_poses[:,2], linewidth=2, color='red')
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        plt.show()

        imgs = utl.get_dji_raw_imgs(data_dir)
        kin = CameraKinematics(test_config["cam_hfov"], vis=True)
        kin.reproject(drone_poses, gt_poses, utl.make_smooth(log_data[:,5:8]), imgs)

    def process_mavic_with_gps_gt(self):

        gt_data = genfromtxt(self._args.processed_gt_file, delimiter=',')
        gt_data[:,3] = 0

        data_dir, _ = os.path.split(self._args.test_config_file)
        log_data = genfromtxt(data_dir+"/log.txt", delimiter=',')

        if not utl.check_time_overlap(log_data[:,1], gt_data[:,0]):
            print("Error! No time overlap found.")
            return

        with open(self._args.test_config_file, 'r') as stream:
            test_config = yaml.safe_load(stream)

        ref_loc = gt_data[0,1:4]

        drone_poses = utl.gps_to_ned(ref_loc, log_data[:,2:5])
        gt_poses = utl.gps_to_ned(ref_loc, gt_data[:,1:4])
        gt_poses, sub_idxs = utl.interpolate(log_data[:,1], gt_data[:,0], \
                                                gt_poses, synced=True)
        drone_poses = drone_poses[sub_idxs]
        euls = log_data[:,5:8]
        euls = euls[sub_idxs]


        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')
        # ax.plot(gt_data[:,0], gt_data[:,1], 0, linewidth=2)
        ax.plot(gt_poses[:,0], gt_poses[:,1], 0, linewidth=2)
        ax.plot(drone_poses[:,0], drone_poses[:,1], drone_poses[:,2], linewidth=2, color='red')
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        plt.show()

        imgs = utl.get_dji_raw_imgs(data_dir)
        kin = CameraKinematics(test_config["cam_hfov"], vis=True)
        print(drone_poses.shape, gt_poses.shape, euls.shape, len(np.array(imgs)[sub_idxs]))
        kin.reproject(drone_poses, gt_poses, utl.make_smooth(euls), np.array(imgs)[sub_idxs])

parser = argparse.ArgumentParser()
parser.add_argument("processed_gt_file", help="The csv file containing target groundtruth.")
parser.add_argument("test_config_file", help="The yaml file containing test setup info.")
args = parser.parse_args()

gtplot = GTPlot(args)
# gtplot.process_mavic_with_camera_gt()
gtplot.process_mavic_with_gps_gt()
