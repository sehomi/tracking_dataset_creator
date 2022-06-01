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
import utils as utl


class Optimizer:

    def __init__(self, path, gt_path, d_type):

        self._tello=False
        if d_type==0:
            self.path = path

            self.img_files = utl.get_dataset_imgs(path)
            self.img_shape = cv.imread(self.img_files[0]).shape
            self._occlusion = genfromtxt(path+"/occlusion.tag", delimiter=',')
            self._gt_boxes = genfromtxt(path+"/groundtruth.txt", delimiter=',')
            self._f = (0.5 * self.img_shape[1] * (1.0 / np.tan((66.0/2.0)*np.pi/180)));
            self.calc_centers()

            gt_data = genfromtxt(gt_path, delimiter=',')
            gt_data[:,3] = 0

            stats_data = genfromtxt(path+"/camera_states.txt", delimiter=',')

            ref_loc = gt_data[0,1:4]

            drone_poses = utl.gps_to_ned(ref_loc, stats_data[:,1:4])
            gt_poses = utl.gps_to_ned(ref_loc, gt_data[:,1:4])
            self._gt_poses, sub_idxs = utl.interpolate(stats_data[:,0], gt_data[:,0], \
                                                    gt_poses, synced=True)
            num_imgs = len(self.img_files)
            sub_idxs = np.array(sub_idxs)
            sub_idxs = sub_idxs[sub_idxs<num_imgs]
            self._gt_poses = self._gt_poses[0:sub_idxs.shape[0]]
            self._drone_poses = drone_poses[sub_idxs]
            self.img_files = np.array(self.img_files)[sub_idxs]
            self._occlusion = self._occlusion[sub_idxs]
            self._gt_boxes = self._gt_boxes[sub_idxs]
            self._centers = self._centers[sub_idxs]
            euls = stats_data[:,4:7]
            self._euls = euls[sub_idxs]
            self._smooth_eul = utl.make_smooth(self._euls)

            self._times = stats_data[:,0][sub_idxs]
            self._times = self._times - self._times[0]

            self.kin = CameraKinematics(66, vis=False)

        if d_type==1:
            self.path = path

            self.img_files = utl.get_dataset_imgs(path)
            self.img_shape = cv.imread(self.img_files[0]).shape
            self._occlusion = genfromtxt(path+"/occlusion.tag", delimiter=',')
            self._gt_boxes = genfromtxt(path+"/groundtruth.txt", delimiter=',')
            self._f = (0.5 * self.img_shape[1] * (1.0 / np.tan((66.0/2.0)*np.pi/180)));
            self.calc_centers()

            gt_data = genfromtxt(gt_path, delimiter=',')
            gt_data = np.insert(gt_data, 2, 0, axis=1)

            stats_data = genfromtxt(path+"/camera_states.txt", delimiter=',')

            with open(path+"/test_config.txt", 'r') as stream:
                test_config = yaml.safe_load(stream)

            bl_loc = test_config["p_bottom_left_ned"]
            angle = test_config["field_angle"]

            drone_poses = utl.gps_to_ned(bl_loc, stats_data[:,1:4])
            gt_poses = utl.interpolate(stats_data[:,0], gt_data[:,3], gt_data[:,:3])
            gt_poses = utl.field_to_ned(gt_poses, angle)

            num_imgs = len(self.img_files)
            num_gt = gt_poses.shape[0]
            num_dt = drone_poses.shape[0]

            limit = np.min([num_dt,num_imgs, num_gt])
            sub_idxs = np.array([i for i in range(limit)])
            # sub_idxs = sub_idxs[sub_idxs<num_imgs]

            self._gt_poses = gt_poses[sub_idxs]
            self._drone_poses = drone_poses[sub_idxs]
            euls = stats_data[:,4:7]
            self._euls = euls[sub_idxs]
            self._smooth_eul = utl.make_smooth(self._euls)
            self._occlusion = self._occlusion[sub_idxs]
            self._gt_boxes = self._gt_boxes[sub_idxs]
            self._centers = self._centers[sub_idxs]

            self._times = stats_data[:,0][sub_idxs]
            self._times = self._times - self._times[0]

            self.kin = CameraKinematics(66, vis=False)

        if d_type==2:
            self.path = path

            self.img_files = utl.get_dataset_imgs(path)
            self.img_shape = cv.imread(self.img_files[0]).shape
            self._occlusion = genfromtxt(path+"/occlusion.tag", delimiter=',')
            self._gt_boxes = genfromtxt(path+"/groundtruth.txt", delimiter=',')
            self._f = (0.5 * self.img_shape[1] * (1.0 / np.tan((55.0/2.0)*np.pi/180)))

            self.calc_centers()

            stats_data = genfromtxt(path+"/camera_states.txt", delimiter=',')

            drone_poses = stats_data[:,1:4]
            drone_poses[:,2] = -drone_poses[:,2]

            ## 0.5 HZ
            # drone_poses[:,2] += 0.03
            # drone_poses[:,1] = -0.13
            ## 2.1 HZ
            drone_poses[:,2] += 0.03
            drone_poses[:,1] = -0.03

            gt_poses = np.zeros((drone_poses.shape[0],3))

            num_imgs = len(self.img_files)
            num_gt = gt_poses.shape[0]
            num_dt = drone_poses.shape[0]


            self._gt_poses = gt_poses
            self._drone_poses = drone_poses
            euls = stats_data[:,4:7]
            self._euls = euls
            self._smooth_eul = utl.make_smooth(self._euls)

            self._times = stats_data[:,0]
            self._times = self._times - self._times[0]

            self.kin = CameraKinematics(55.0, vis=False)
            self._tello=True

        fig_3d=plt.figure(0)
        self.ax_3d=plt.axes(projection ='3d')
        self.ax_3d.set_title('Trajectories Plot')


    def calc_centers(self):
        centers = []
        for gt in self._gt_boxes:
            x = int(gt[0])
            y = int(gt[1])
            w = int(gt[2] - x)
            h = int(gt[5] - y)
            centers.append( [(x+w/2)/self._f, (y+h/2)/self._f] )

        self._centers = np.array(centers)


    def calc_boxes(self, xns):

        open(self.path+"/groundtruth_corrected.txt", "w").close()

        boxes = []
        for i, ct in enumerate(xns):
            w = int(self._gt_boxes[i,2] - self._gt_boxes[i,0])
            h = int(self._gt_boxes[i,5] - self._gt_boxes[i,1])
            if self._occlusion[i]:
                rect = (ct[0] - int(w/2), ct[1] - int(h/2), w, h)
            else:
                rect = (int(self._centers[i][0]*self._f)-int(w/2),int(self._centers[i][1]*self._f)-int(h/2),w,h)

            boxes.append(rect)

            x1 = rect[0]
            y1 = rect[1]
            x2 = rect[0] + rect[2]
            y2 = rect[1]
            x3 = rect[0] + rect[2]
            y3 = rect[1] + rect[3]
            x4 = rect[0]
            y4 = rect[1] + rect[3]

            f = open(self.path+"/groundtruth_corrected.txt", "a")
            f.write("{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(x1,y1,x2,y2,x3,y3,x4,y4))
            f.close()

        return boxes


    def plot_trajs(self, wait=True):

        self.ax_3d.cla()

        if wait:
            self.ax_3d.plot(self.new_poses[:1000,0],self.new_poses[:1000,1],self.new_poses[:1000,2], color='blue')
            self.ax_3d.plot(self._gt_poses[:1000,0],self._gt_poses[:1000,1],self._gt_poses[:1000,2], color='black')

            deriv = 5*utl.compare_curves(self._gt_poses, self.new_poses)
            # self.ax_3d.quiver(self._gt_poses[0:1000:50,0], self._gt_poses[0:1000:50,1], \
            #                   self._gt_poses[0:1000:50,2], deriv[0:1000:50,0], \
            #                   deriv[0:1000:50,1], deriv[0:1000:50,2], length=1, \
            #                   linewidth=2, color='red', pivot='tail')
        else:
            self.ax_3d.plot(self._poses_list[:,0],self._poses_list[:,1],self._poses_list[:,2], color='green')

        self.ax_3d.set_xlabel('x')
        self.ax_3d.set_ylabel('y')
        self.ax_3d.set_zlabel('z')

        # self.ax_3d.set_xlim(-30,20)
        # self.ax_3d.set_ylim(0,50)
        # ax.set_zlim(7,12)

        utl.set_axes_equal(self.ax_3d)

        if wait:
            plt.show()
        else:
            plt.pause(0.03)


    def G_Y(self, Y):
        x = Y[0]
        y = Y[1]
        z = Y[2]

        jac = np.zeros((2,3))
        jac[0,0] = self._f/z
        jac[0,1] = 0
        jac[0,2] = -x*self._f/z**2
        jac[1,0] = 0
        jac[1,1] = self._f/z
        jac[1,2] = -y*self._f/z**2
        return jac



    def optimize(self):

        # self._smooth_eul = utl.time_shift(self._times, self._smooth_eul, 0.2)
        # self._drone_poses = utl.time_shift(self._times, self._drone_poses, 0.2)
        # self._smooth_eul = utl.shift(self._smooth_eul, -5)

        print("Optimizing non occluded frames...")

        self.new_poses = self._gt_poses.copy()

        if not self._tello:
            errs = np.zeros((self._gt_poses.shape[0],3))

            rate_coef = 0.001
            slowdown=False
            poses_list=[]
            for j in range(self._gt_poses.shape[0]):
                if self._occlusion[j]: continue

                beta=0
                if self._tello:
                    beta=12
                Tcb = utl.make_DCM([(90-beta)*np.pi/180, 0, 90*np.pi/180])
                Tbi = utl.make_DCM(self._smooth_eul[j,:])
                T = np.matmul(Tcb, Tbi)
                X = self._gt_poses[j,:].copy()
                center = self._centers[j]*self._f - np.array([self.img_shape[1]/2, self.img_shape[0]/2])

                counter=0
                while True:
                    xn_est = self.kin.reproject_single(self._drone_poses[j], X, \
                                                       self._smooth_eul[j], self.img_shape, \
                                                       tello=self._tello)
                    xn_est -= np.array([self.img_shape[1]/2, self.img_shape[0]/2])

                    Y = np.matmul( T, (X - self._drone_poses[j]) )

                    grad = np.matmul( 2*(xn_est - center) , np.matmul(self.G_Y(Y) , T) )
                    grad = rate_coef*grad

                    X[0] = X[0] - grad[0]
                    X[1] = X[1] - grad[1]

                    if np.linalg.norm(grad) < 0.1:
                        break

                    if counter > 200:
                        rate_coef *= 0.1
                        counter=0
                        slowdown=True

                    counter += 1

                if slowdown:
                    rate_coef = 0.001
                    slowdown=False

                # if np.linalg.norm(self.new_poses[j,:] - X) > 20:
                #     self._occlusion[j] = True
                #     continue

                self.new_poses[j,:] = X
                errs[j,:] = X - self._gt_poses[j,:]

                first_img = cv.imread(self.img_files[j])
                rect = (int(xn_est[0]+self.img_shape[1]/2)-1,int(xn_est[1]+self.img_shape[0]/2)-1,2,2)
                cv.rectangle(first_img, rect, (0,255,255), 2)
                rect = (int(self._centers[j][0]*self._f)-1,int(self._centers[j][1]*self._f)-1,2,2)
                cv.rectangle(first_img, rect, (0,0,255), 2)
                cv.imshow("image", first_img)
                cv.waitKey(3)

            print("Interpolating for occluded frames...")

            last_err = None
            last_err_idx = None
            next_err = None
            for j in range(self._gt_poses.shape[0]):
                if self._occlusion[j] and last_err is None:
                    last_err = errs[j-1]
                    next_err = None
                    last_err_idx = j

                if not self._occlusion[j] and next_err is None and last_err is not  None:
                    next_err = errs[j]

                    length = j - last_err_idx
                    for k in range(last_err_idx, j):
                        err = (k - last_err_idx)/length * next_err + \
                              (j - k)/length * last_err
                        self.new_poses[k,:] = self._gt_poses[k,:] + err

                    last_err = None

            self.new_poses = utl.make_smooth(self.new_poses)
            # self.plot_trajs()

        poses_list=[]
        xn_ests = []
        for j in range(len(self.new_poses)):
            xn_est = self.kin.reproject_single(self._drone_poses[j], self.new_poses[j], \
                                               self._smooth_eul[j], self.img_shape, \
                                               tello=self._tello)
            xn_ests.append(xn_est)
            # print(self._drone_poses[j], self.new_poses[j])
            # print(xn_est)

        boxes = self.calc_boxes(xn_ests)
        for j, box in enumerate(boxes):
            img = cv.imread(self.img_files[j])

            if self._occlusion[j]:
                cv.rectangle(img, box, (0,255,255), 2)
            else:
                cv.rectangle(img, box, (0,0,255), 2)

            cv.imshow("image", img)
            cv.waitKey(33)

            # poses_list.append(self._drone_poses[j,:])
            # poses_list.append(self.new_poses[j])
            # self._poses_list = np.array(poses_list)
            # self.plot_trajs(wait=False)

        self.plot_trajs()


    def show(self):

        self.optimize()
        # self.optimize_single_point()
        for i, img in enumerate(self.img_files):
            gt = self._gt_boxes[i, :]
            image = cv.imread(img)

            x = int(gt[0])
            y = int(gt[1])
            w = int(gt[2] - x)
            h = int(gt[5] - y)

            rect = (x,y,w,h)

            cv.rectangle(image, rect, (0,255,255), 2)

            cv.imshow("gt", image)
            cv.waitKey(33)

parser = argparse.ArgumentParser()
parser.add_argument("folder_path", help="Path to data folder")
parser.add_argument("gt_path", help="Path to target groundtruth positions")
parser.add_argument("data_type", help="Data type is one of the following: 0 for mavic with gps groundtruth, \
                                                                          1 for mavic with camera groundtruth,\
                                                                          2 for tello in static configuration")
args = parser.parse_args()

opt = Optimizer(args.folder_path, args.gt_path, int(args.data_type))
opt.show()
