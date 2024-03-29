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


    def to3D(self, visualize=False):

        self.new_poses = self._gt_poses.copy()

        self._gt_poses_with_range = []
        self._gt_poses_with_range_csen = []
        self._gt_poses_with_range_gcn = []
        self._gt_poses_level = []
        for i in tqdm(range(self._gt_boxes.shape[0])):
            box = self._gt_boxes[i].copy()
            direction = self._gt_poses_with_alt[i]-self._drone_poses[i]
            direction = direction/np.linalg.norm(direction)
            self._gt_poses_with_range.append( self._re.findPos(self._img_files[i], box, direction, self._drone_poses[i,2]) + self._drone_poses[i] )
            self._gt_poses_level.append( self._re_level.findPos(self._img_files[i], box, direction, self._drone_poses[i,2]) + self._drone_poses[i] )
            self._gt_poses_with_range_csen.append( self._re_csen.findPos(self._img_files[i], box, direction, self._drone_poses[i,2]) + self._drone_poses[i] )
            self._gt_poses_with_range_gcn.append( self._re_gcn.findPos(self._img_files[i], box, direction, self._drone_poses[i,2]) + self._drone_poses[i] )

        self._gt_poses_with_range = np.array(self._gt_poses_with_range)
        self._gt_poses_level = np.array(self._gt_poses_level)
        self._gt_poses_with_range_csen = np.array(self._gt_poses_with_range_csen)
        self._gt_poses_with_range_gcn = np.array(self._gt_poses_with_range_gcn)

        self.dump_to_file()

        if visualize:
            # self.plot_trajs_2d()
            # self.plot_trajs()
            self.plot_range_errs()

    def dump_to_file(self):

      # df = pd.DataFrame({'level' : self._gt_poses_level, 'fcn' : self._gt_poses_with_range,
      #                    'csen': self._gt_poses_with_range_csen, 'gcn': self._gt_poses_with_range_gcn}, index=0)
      # name = os.path.basename(os.path.normpath(self.path))
      # df.to_csv("{}.csv".format(name), index=False)

      con = np.concatenate((self._gt_poses_level, self._gt_poses_with_range,\
                            self._gt_poses_with_range_csen, self._gt_poses_with_range_gcn), axis=1)
      name = os.path.basename(os.path.normpath(self.path))
      np.savetxt("{}.csv".format(name), con, delimiter=",")

    def read_from_file(self, visualize=True):

      # name = os.path.basename(os.path.normpath(self.path))
      # df = pd.read_csv("{}.csv".format(name))

      name = os.path.basename(os.path.normpath(self.path))
      data = np.genfromtxt("{}.csv".format(name), delimiter=',')

      self._gt_poses_with_range = data[:,3:6].copy()
      self._gt_poses_with_range_csen = data[:,6:9].copy()
      self._gt_poses_with_range_gcn = data[:,9:12].copy()
      self._gt_poses_level = data[:,0:3].copy()

      # print(self._gt_poses_level)

      if visualize:
            # self.plot_trajs_2d()
            # self.plot_trajs()
            self.plot_range_errs()

parser = argparse.ArgumentParser()
parser.add_argument("folder_path", help="Path to data folder")
parser.add_argument("gt_path", help="Path to target groundtruth positions")
parser.add_argument("csen_path", help="Path to CSENDistance package")
parser.add_argument("gcn_path", help="Path to GCNDepth package")

args = parser.parse_args()

opt = Position3DVisualizer(args.folder_path, args.gt_path, plot_type='ERR', \
                           csen_pkg_path=args.csen_path , gcn_pkg_path=args.gcn_path)
# opt.to3D()
opt.read_from_file(visualize=True)

exit()

ests = {'act':[],'level':[],'fcn':[],'csen':[],'gcn':[]}

for i in range(1,8):

    folder_path = list(args.folder_path)
    gt_path = list(args.gt_path)

    num_idx = args.folder_path.rfind('/')-1
    folder_path[num_idx] = str(i)

    num_idx = args.gt_path.find('.txt')-1
    gt_path[num_idx] = str(i)

    folder_path = ''.join(folder_path)
    gt_path = ''.join(gt_path)
    print(folder_path, gt_path)
    opt = Position3DVisualizer(folder_path, gt_path, plot_type='ERR', \
                              csen_pkg_path=args.csen_path , gcn_pkg_path=args.gcn_path)
    # opt.to3D()
    opt.read_from_file(visualize=True)

    ests['act'].append(
      np.linalg.norm(opt._gt_poses_with_alt[:]-opt._drone_poses[:], axis=1)
    )
    ests['level'].append(
      np.linalg.norm(opt._gt_poses_level[:]-opt._drone_poses[:], axis=1)
    )
    ests['fcn'].append(
      np.linalg.norm(opt._gt_poses_with_range[:]-opt._drone_poses[:], axis=1)
    )
    ests['csen'].append(
      np.linalg.norm(opt._gt_poses_with_range_csen[:]-opt._drone_poses[:], axis=1)
    )
    ests['gcn'].append(
      np.linalg.norm(opt._gt_poses_with_range_gcn[:]-opt._drone_poses[:], axis=1)
    )

    del opt

def scatter_plot(act, est, name):
  plt.rcParams["figure.figsize"] = (5,5)
  fig, ax = plt.subplots()

  ax.plot([0,30], [0,30], color='red', linewidth=2)

  act = np.hstack(act)
  est = np.hstack(est)

  rmv_idxs = np.where(est>49)
  est = np.delete(est, rmv_idxs) 
  act = np.delete(act, rmv_idxs) 
  
  sort_idxs = np.argsort( np.abs(est-act) )
  
  sort_idxs = np.delete(sort_idxs, np.random.choice(range(1000), size=900, replace=False)) 
  est = est[sort_idxs]
  act = act[sort_idxs]

  ax.scatter(act, est, color='blue', s=4)\

  rms = np.sqrt(np.mean((est-act)**2))
  ax.fill_between(np.arange(0,31), np.arange(0,31)+rms, np.arange(0,31)-rms, color='red', alpha=0.3)

  ax.legend()
  ax.grid()

  ax.arrow(x=15, y=15, dx=0, dy=rms, color= 'black', linewidth=2.5, head_width=0.3)
  ax.text(16, 20, 'RMS = {:.1f}'.format(rms), color= 'black', weight='bold')

  ax.set_xlabel('Actual Distance (m)')
  ax.set_ylabel('Estimated Distance (m)')

  ax.set_xlim([0,30])
  ax.set_ylim([0,30])

  # name = os.path.basename(os.path.normpath(path))
  plt.savefig('scatter_{}.pdf'.format(name), format='pdf')


# scatter_plot(ests['act'], ests['level'], 'level')
# scatter_plot(ests['act'], ests['fcn'], 'fcn')
# scatter_plot(ests['act'], ests['csen'], 'csen')
scatter_plot(ests['act'], ests['gcn'], 'gcn')

