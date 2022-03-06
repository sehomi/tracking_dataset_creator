#!/usr/bin/env python

import numpy as np
from numpy import genfromtxt
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import os
import utm

def gps_to_field(bl_loc, angle, locs):

    angle_rad = angle * 3.1415 / 180.0

    DCM = np.zeros((3,3))
    DCM[0,0] = np.cos(angle_rad)
    DCM[0,1] = np.sin(angle_rad)
    DCM[1,0] = -np.sin(angle_rad)
    DCM[1,1] = np.cos(angle_rad)
    DCM[2,2] = 1

    y_bl, x_bl, _, _ = utm.from_latlon(bl_loc[0], bl_loc[1])
    pose_bl_utm = np.array( [x_bl, y_bl, -bl_loc[2]] )

    poses = []
    for loc in locs:
        y, x, _, _ = utm.from_latlon(loc[0], loc[1])
        pose_utm = [x,y,-loc[2]]
        pose = np.matmul(DCM,pose_utm-pose_bl_utm)
        poses.append(pose)

    poses = np.array(poses)
    return poses



parser = argparse.ArgumentParser()
parser.add_argument("processed_gt_file", help="The csv file containing target groundtruth.")
parser.add_argument("test_config_file", help="The yaml file containing test setup info.")
args = parser.parse_args()


gt_data = genfromtxt(args.processed_gt_file, delimiter=',')

data_dir, _ = os.path.split(args.test_config_file)
log_data = genfromtxt(data_dir+"/log.txt", delimiter=',')
locs = log_data[:,2:5]

with open(args.test_config_file, 'r') as stream:
    test_config = yaml.safe_load(stream)

bl_loc = test_config["p_bottom_left_ned"]
angle = test_config["field_angle"]

drone_poses = gps_to_field(bl_loc, angle, locs)

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')

ax.plot(gt_data[:,0], gt_data[:,1], 0, linewidth=2)
ax.plot(drone_poses[:,0], drone_poses[:,1], drone_poses[:,2], linewidth=2, color='red')

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")

plt.show()
