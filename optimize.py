#!/usr/bin/env python

import os
import numpy as np
from numpy import genfromtxt
import argparse
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares, differential_evolution

from camera_kinematics import CameraKinematics
import utils as utl


class Optimizer:

    def __init__(self, path, gt_path, log_path):

        self.img_files = utl.get_dataset_imgs(path)
        self.img_shape = cv.imread(self.img_files[0]).shape

        self._occlusion = genfromtxt(path+"/occlusion.tag", delimiter=',')
        print(self._occlusion)

        self._gt_boxes = genfromtxt(path+"/groundtruth.txt", delimiter=',')
        self._f = (0.5 * 440 * (1.0 / np.tan((66.0/2.0)*np.pi/180)));
        self.calc_centers()

        gt_data = genfromtxt(gt_path, delimiter=',')
        gt_data[:,3] = 0

        log_data = genfromtxt(log_path, delimiter=',')

        ref_loc = gt_data[0,1:4]

        drone_poses = utl.gps_to_ned(ref_loc, log_data[:,2:5])
        gt_poses = utl.gps_to_ned(ref_loc, gt_data[:,1:4])
        self._gt_poses, sub_idxs = utl.interpolate(log_data[:,1], gt_data[:,0], \
                                                gt_poses, synced=True)
        self._drone_poses = drone_poses[sub_idxs]
        euls = log_data[:,5:8]
        self._euls = euls[sub_idxs]
        self._smooth_eul = utl.make_smooth(self._euls)

        self._times = log_data[:,1][sub_idxs]
        self._times = self._times - self._times[0]

        self.kin = CameraKinematics(66, vis=False)


    def linearize(self, i):

        # A = []
        # b = []
        # for i in range(self._euls.shape[0]):
        #     R = utl.make_DCM(self._euls[i,:])
        #
        #     Xc =  np.matmul(R, self._gt_poses[i,:] - self._drone_poses[i,:])
        #
        #     xc = Xc[0]
        #     yc = Xc[1]
        #     zc = Xc[2]
        #
        #     G = np.zeros((2,3))
        #     G[0,0] = 1/zc
        #     G[0,1] = 0
        #     # G[0,2] = -xc/(zc**2)
        #     G[0,2] = 0
        #     G[1,0] = 0
        #     G[1,1] = 1/zc
        #     # G[1,2] = -yc/(zc**2)
        #     G[1,2] = 0
        #
        #     G = np.matmul(G,R)
        #     A.append(G)
        #
        #     Xn_est = np.array([xc/zc, yc/zc])
        #     b.append(self._centers[i] - Xn_est)
        #
        # return np.array(A), np.array(b)

        R = utl.make_DCM(self._euls[i,:])
        Xc =  np.matmul(R, self._gt_poses[i,:] - self._drone_poses[i,:])

        xc = Xc[0]
        yc = Xc[1]
        zc = Xc[2]

        G = np.zeros((2,3))
        G[0,0] = 1/zc
        G[0,1] = 0
        # G[0,2] = -xc/(zc**2)
        G[0,2] = 0
        G[1,0] = 0
        G[1,1] = 1/zc
        # G[1,2] = -yc/(zc**2)
        G[1,2] = 0

        G = np.matmul(G,R)

        Xn_est = np.array([xc/zc, yc/zc])
        b = self._centers[i] - Xn_est

        return G,b

    def calc_centers(self):
        centers = []
        for gt in self._gt_boxes:
            x = int(gt[0])
            y = int(gt[1])
            w = int(gt[2] - x)
            h = int(gt[5] - y)
            centers.append( [(x+w/2)/self._f, (y+h/2)/self._f] )

        self._centers = np.array(centers)

    def solve(self, A, b):

        delta = []
        for i, G in enumerate(A):
            try:
                G_t = np.transpose(G)
                d = np.linalg.inv( np.matmul(G_t, G) )
                d = np.matmul(d, G_t)
                d = np.matmul(d, b[i])
                delta.append(d)
                # print(d)
            except:
                delta.append([0,0,0])
                # print("singularity")

        delta = np.array(delta)
        return delta

    # def loss(self, A, b, d):
    #
    #     loss = 0
    #     for i, G in enumerate(A):
    #         loss = loss + np.linalg.norm(np.matmul(G, d[i]) - b[i])**2
    #
    #     return loss

    def loss(self, X):

        new_pos = self._gt_poses[self._i]
        new_pos[0] = X[0]
        new_pos[1] = X[1]

        xn_est = self.kin.reproject_single(self._drone_poses[self._i], new_pos, self._smooth_eul[self._i], self.img_shape)
        xn_est = np.array( [val/self._f for val in xn_est] )

        loss = np.linalg.norm(xn_est*self._f - self._centers[self._i]*self._f)**2
        print(loss)

        # first_img = cv.imread(self.img_files[self._i])
        # rect = (int(xn_est[0]*self._f)-1,int(xn_est[1]*self._f)-1,2,2)
        # cv.rectangle(first_img, rect, (0,255,255), 2)
        # rect = (int(self._centers[self._i][0]*self._f)-1,int(self._centers[self._i][1]*self._f)-1,2,2)
        # cv.rectangle(first_img, rect, (0,0,255), 2)
        # cv.imshow("image", first_img)
        # cv.waitKey(33)

        return loss


    def loss1(self, X):

        print(X)

        new_poses = self._gt_poses.copy()
        loss = 0
        # for j in range(new_poses.shape[0]):
        #     if self._occlusion[j]: continue
        #     if j%10!=0: continue
        j=self._i
        new_pos = [new_poses[j][0], new_poses[j][1], new_poses[j][2]]

        t = self._times[j]

        for k in range(self._deg):
            new_pos[0] += X[k+0*self._deg]*t**k
            new_pos[1] += X[k+1*self._deg]*t**k
            # new_pos[2] += X[k+2*self._deg]*t**k

        new_poses[j] = new_pos

        xn_est = self.kin.reproject_single(self._drone_poses[j], new_pos, self._smooth_eul[j], self.img_shape)
        xn_est = np.array( [val/self._f for val in xn_est] )

        loss += np.linalg.norm(xn_est*self._f - self._centers[j]*self._f)**2

            # img = cv.imread(self.img_files[j])
            # cv.imshow("test", img)
            # cv.waitKey(1)


        self.new_poses = np.array(new_poses)
        # self.plot_trajs()

        print(loss)

        xn_est = self.kin.reproject_single(self._drone_poses[self._i], new_poses[self._i], utl.make_smooth(self._euls)[self._i], self.img_shape)
        xn_est = np.array( [val/self._f for val in xn_est] )

        first_img = cv.imread(self.img_files[self._i])
        rect = (int(xn_est[0]*self._f)-1,int(xn_est[1]*self._f)-1,2,2)
        cv.rectangle(first_img, rect, (0,255,255), 2)
        rect = (int(self._centers[self._i][0]*self._f)-1,int(self._centers[self._i][1]*self._f)-1,2,2)
        cv.rectangle(first_img, rect, (0,0,255), 2)
        cv.imshow("image", first_img)
        cv.waitKey(1)

        return loss

    def plot_trajs(self):
        fig_3d=plt.figure(0)
        ax_3d=plt.axes(projection ='3d')
        ax_3d.set_title('Trajectories Plot')

        ax_3d.cla()

        ax_3d.plot(self.new_poses[:,0],self.new_poses[:,1],self.new_poses[:,2], color='blue')
        ax_3d.plot(self._gt_poses[:,0],self._gt_poses[:,1],self._gt_poses[:,2], color='black')

        ax_3d.set_xlabel('x')
        ax_3d.set_ylabel('y')
        ax_3d.set_zlabel('z')

        plt.show()

    def f(self, X):
        x = X[0]
        y = X[1]
        z = X[2]

        return np.array( [x*self._f/z, y*self._f/z] ).reshape((2,1))

    def f_p(self, Y):
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

        self._i=1000
        j=self._i
        Tcb = utl.make_DCM([90*np.pi/180, 0, 90*np.pi/180])
        Tbi = utl.make_DCM(self._euls[j,:])
        T = np.matmul(Tcb, Tbi)
        X = self._gt_poses[j,:]
        center = self._centers[j]*self._f - np.array([self.img_shape[1]/2, self.img_shape[0]/2])

        for i in range(1000):
            xn_est = self.kin.reproject_single(self._drone_poses[self._i], X, self._smooth_eul[self._i], self.img_shape)
            xn_est -= np.array([self.img_shape[1]/2, self.img_shape[0]/2])
            # print(xn_est)

            Y = np.matmul( T, (X - self._drone_poses[self._i]) )

            # Y = Y/np.linalg.norm(Y)
            # print(Y)
            # center_est = self.kin.from_direction_vector(Y, self.kin._cx, self.kin._cy, self.kin._f)
            # rect = (center_est[0]-1,center_est[1]-1,2,2)


            grad = np.matmul( 2*(xn_est - center) , np.matmul(self.f_p(Y) , T) )
            grad = 0.001*grad
            # print("\n\n*****\n\n")
            # print(grad)
            # print(2*(xn_est - center))
            # print(self.f_p(Y))
            # print(T)
            X[0] = X[0] - grad[0]
            X[1] = X[1] - grad[1]



            first_img = cv.imread(self.img_files[self._i])
            rect = (int(xn_est[0]+self.img_shape[1]/2)-1,int(xn_est[1]+self.img_shape[0]/2)-1,2,2)
            cv.rectangle(first_img, rect, (0,255,255), 2)
            rect = (int(self._centers[self._i][0]*self._f)-1,int(self._centers[self._i][1]*self._f)-1,2,2)
            # cv.rectangle(first_img, rect, (0,0,255), 2)
            cv.imshow("image", first_img)
            cv.waitKey(33)

        # X = self._gt_poses
        #
        # for i in range(1000):
        #     A, b = self.linearize()
        #     de = self.solve(A,b)
        #     self._gt_poses[:51] = self._gt_poses[:51] + de
        #     print(self.loss(A,b,de))
        #     print(de[30])

    def optimize1(self):

        # X = self._gt_poses
        #
        # for i in range(1000):
        #     A, b = self.linearize()
        #     de = self.solve(A,b)
        #     self._gt_poses[:51] = self._gt_poses[:51] + de
        #     print(self.loss(A,b,de))
        #     print(de[30])
        #
        #     self.kin.reproject(self._drone_poses, self._gt_poses, utl.make_smooth(self._euls), self.img_files)

        # i=1000
        #
        # de = np.zeros((3,1))
        # for j in range(1000):
        #     print(self._gt_poses[i])
        #     G, b = self.linearize(i)
        #     t = np.matmul(G,de) - b.reshape((2,1))
        #     de = de + 0.2*np.matmul(G.T,t)
        #     new_pos = self._gt_poses[i].reshape((3,1)) + de
        #     self._gt_poses[i,0] = new_pos[0]
        #     self._gt_poses[i,1] = new_pos[1]
        #     self._gt_poses[i,2] = new_pos[2]
        #
        #     self.kin.reproject_single(self._drone_poses[i], self._gt_poses[i], utl.make_smooth(self._euls)[i], self.img_files[i])


        # i=50
        #
        # de = np.zeros((3,1))
        # for j in range(1000):
        #     # print(self._gt_poses[i])
        #     G, b = self.linearize(i)
        #
        #     xn_est = self.kin.reproject_single(self._drone_poses[i], self._gt_poses[i], utl.make_smooth(self._euls)[i], self.img_files[i])
        #     xn_est = np.array( [val/self._f for val in xn_est] )
        #     t = xn_est - self._centers[i]
        #
        #     diff = 2*np.matmul(G.T,t)
        #     # new_pos = self._gt_poses[i] +
        #     print(diff)
        #     self._gt_poses[i,0] += diff[1]
        #     self._gt_poses[i,1] -= diff[0]
        #     self._gt_poses[i,2] += diff[2]
        #
        #     first_img = cv.imread(self.img_files[i])
        #     rect = (int(xn_est[0]*self._f)-1,int(xn_est[1]*self._f)-1,2,2)
        #     cv.rectangle(first_img, rect, (0,255,255), 2)
        #     rect = (int(self._centers[i][0]*self._f)-1,int(self._centers[i][1]*self._f)-1,2,2)
        #     cv.rectangle(first_img, rect, (0,0,255), 2)
        #     cv.imshow("image", first_img)
        #     cv.waitKey()

        # self._i=50
        # self._deg = 5
        # X0 = np.array([self._gt_poses[self._i,0], self._gt_poses[self._i,1]])
        # result = least_squares(self.loss, X0, diff_step=0.1, max_nfev=100000)
        # print(result)

        # for i in range(50, 2000):
        #     if self._occlusion[i]: continue
        #
        #     self._i=i
        #     self._deg = 5
        #     X0 = np.array([self._gt_poses[self._i,0], self._gt_poses[self._i,1]])
        #     result = least_squares(self.loss, X0, diff_step=0.1, max_nfev=100000)
        #     print(result)
        #
        #     new_pos = self._gt_poses[self._i]
        #     new_pos[0] = result.x[0]
        #     new_pos[1] = result.x[1]
        #
        #     xn_est = self.kin.reproject_single(self._drone_poses[self._i], new_pos, self._smooth_eul[self._i], self.img_shape)
        #     xn_est = np.array( [val/self._f for val in xn_est] )
        #
        #     first_img = cv.imread(self.img_files[self._i])
        #     rect = (int(xn_est[0]*self._f)-1,int(xn_est[1]*self._f)-1,2,2)
        #     cv.rectangle(first_img, rect, (0,255,255), 2)
        #     rect = (int(self._centers[self._i][0]*self._f)-1,int(self._centers[self._i][1]*self._f)-1,2,2)
        #     cv.rectangle(first_img, rect, (0,0,255), 2)
        #     cv.imshow("image", first_img)
        #     cv.waitKey(33)

        self._i=50
        self._deg = 2
        X0 = np.zeros( (1,2*self._deg) )
        # X0 = np.array( [[3, -6]] )
        print(X0[0])
        result = least_squares(self.loss1, X0[0], diff_step=10, max_nfev=100000)
        # XL = -10*np.ones( (1,2*self._deg) )
        # XU = 10*np.ones( (1,2*self._deg) )
        # bounds = [(XL[0][i],XU[0][i]) for i in range(2*self._deg)]
        # result = differential_evolution(self.loss1, bounds=bounds, maxiter=10000000, popsize=20,tol=1e-6)
        self.plot_trajs()
        print(result)

    def show(self):

        self.optimize()
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
parser.add_argument("log_path", help="DJI log path")
args = parser.parse_args()

opt = Optimizer(args.folder_path, args.gt_path, args.log_path)
opt.show()
