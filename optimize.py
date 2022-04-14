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
        num_imgs = len(self.img_files)
        sub_idxs = np.array(sub_idxs)
        sub_idxs = sub_idxs[sub_idxs<num_imgs]
        self._gt_poses = self._gt_poses[0:sub_idxs.shape[0]]
        self._drone_poses = drone_poses[sub_idxs]
        self.img_files = np.array(self.img_files)[sub_idxs]
        self._occlusion = self._occlusion[sub_idxs]
        self._gt_boxes = self._gt_boxes[sub_idxs]
        self._centers = self._centers[sub_idxs]
        euls = log_data[:,5:8]
        self._euls = euls[sub_idxs]
        self._smooth_eul = utl.make_smooth(self._euls)

        self._times = log_data[:,1][sub_idxs]
        self._times = self._times - self._times[0]

        self.kin = CameraKinematics(66, vis=False)

        fig_3d=plt.figure(0)
        self.ax_3d=plt.axes(projection ='3d')
        self.ax_3d.set_title('Trajectories Plot')

        print("gt: ", self._gt_poses.shape)
        print("drone: ", self._drone_poses.shape)
        print("eul: ", self._smooth_eul.shape)
        print("imgs: ", len(self.img_files))
        print("occs: ", self._occlusion.shape)
        print("centers: ", self._centers.shape)
        # exit()


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

    def plot_trajs(self, wait=True):

        self.ax_3d.cla()

        self.ax_3d.plot(self.new_poses[:,0],self.new_poses[:,1],self.new_poses[:,2], color='blue')
        self.ax_3d.plot(self._gt_poses[:,0],self._gt_poses[:,1],self._gt_poses[:,2], color='black')
        # self.ax_3d.plot(self._poses_list[:,0],self._poses_list[:,1],self._poses_list[:,2], color='green')

        self.ax_3d.set_xlabel('x')
        self.ax_3d.set_ylabel('y')
        self.ax_3d.set_zlabel('z')

        # self.ax_3d.set_xlim(-10,90)
        # self.ax_3d.set_ylim(-40,10)
        # ax.set_zlim(7,12)

        if wait:
            plt.show()
        else:
            plt.pause(0.03)

    def f(self, X):
        x = X[0]
        y = X[1]
        z = X[2]

        return np.array( [x*self._f/z, y*self._f/z] ).reshape((2,1))

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

    def Ts(self, t):

        ts = np.zeros((self.err_deg,1))
        for c in range(self.err_deg):
            ts[c] = t**c

        return ts

    def X_P(self, t):

        jac = np.zeros((3,3,self.err_deg))

        for i in range(3):
            for j in range(self.err_deg):
                jac[i, i, j] = t**j

        return jac

    def X_P_cos(self, t):

        jac = np.zeros((3,3,2*self.err_deg))

        for i in range(3):
            for j in range(self.err_deg):
                idx = 2*j
                jac[i, i, idx] = np.cos(2*(j+1)*np.pi*t/self._L)
                jac[i, i, idx+1] = np.sin(2*(j+1)*np.pi*t/self._L)

        return jac

    def COSs(self, t):

        jac = np.zeros((2*self.err_deg,1))

        for j in range(self.err_deg):
            idx = 2*j
            jac[idx, 0] = np.cos(2*(j+1)*np.pi*t/self._L)
            jac[idx+1, 0] = np.sin(2*(j+1)*np.pi*t/self._L)

        return jac

    def optimize_single_point(self):

        self._i=1203
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


            grad = np.matmul( 2*(xn_est - center) , np.matmul(self.G_Y(Y) , T) )
            grad = 0.001*grad
            # print("\n\n*****\n\n")
            # print(grad)
            # print(2*(xn_est - center))
            # print(self.f_p(Y))
            # print(T)
            X[0] = X[0] - grad[0]
            X[1] = X[1] - grad[1]

            print(np.linalg.norm(grad))

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

    def optimize(self):

        self.new_poses = self._gt_poses.copy()
        errs = np.zeros((self._gt_poses.shape[0],3))

        rate_coef = 0.001
        slowdown=False
        poses_list=[]
        for j in range(self._gt_poses.shape[0]):
        # for j in range(1000):
            if self._occlusion[j]: continue
            # print(j)
            # self._i=1000
            # j=self._i
            Tcb = utl.make_DCM([90*np.pi/180, 0, 90*np.pi/180])
            Tbi = utl.make_DCM(self._euls[j,:])
            T = np.matmul(Tcb, Tbi)
            X = self._gt_poses[j,:].copy()
            center = self._centers[j]*self._f - np.array([self.img_shape[1]/2, self.img_shape[0]/2])

            counter=0
            while True:
                xn_est = self.kin.reproject_single(self._drone_poses[j], X, self._smooth_eul[j], self.img_shape)
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

            if slowdown and counter<50:
                rate_coef *= 10
                slowdown=False

            self.new_poses[j,:] = X
            # print(errs.shape)
            errs[j,:] = X - self._gt_poses[j,:]
            first_img = cv.imread(self.img_files[j])
            rect = (int(xn_est[0]+self.img_shape[1]/2)-1,int(xn_est[1]+self.img_shape[0]/2)-1,2,2)
            cv.rectangle(first_img, rect, (0,255,255), 2)
            rect = (int(self._centers[j][0]*self._f)-1,int(self._centers[j][1]*self._f)-1,2,2)
            cv.rectangle(first_img, rect, (0,0,255), 2)
            cv.imshow("image", first_img)
            cv.waitKey(3)

            # poses_list.append(self._drone_poses[j,:])
            # poses_list.append(self._gt_poses[j,:])
            # self._poses_list = np.array(poses_list)
            # self.plot_trajs(wait=False)

        last_err = None
        last_err_idx = None
        next_err = None
        for j in range(1000):
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
        self.plot_trajs()



    def optimize_params(self):

        sel = 50
        err_deg = 8
        self.err_deg = err_deg
        self._L = self._times[-1]

        # P = np.zeros((3,err_deg))
        P = np.zeros((3,2*err_deg))
        self.new_poses = self._gt_poses.copy()
        for i in range(1000):
            grad = np.zeros((3,2*err_deg))
            # for j in range(self._gt_poses.shape[0]):
            for j in np.arange(50,1200, 100):
                if self._occlusion[j]: continue

                Tcb = utl.make_DCM([90*np.pi/180, 0, 90*np.pi/180])
                Tbi = utl.make_DCM(self._euls[j,:])
                T = np.matmul(Tcb, Tbi)

                # ts = self.Ts(self._times[j])
                ts = self.COSs(self._times[j])
                # print(np.matmul(P, ts).T[0])
                X = self._gt_poses[j,:] + np.matmul(P, ts).T[0]

                center = self._centers[j]*self._f - np.array([self.img_shape[1]/2, self.img_shape[0]/2])

                xn_est = self.kin.reproject_single(self._drone_poses[j], X, self._smooth_eul[j], self.img_shape)
                xn_est -= np.array([self.img_shape[1]/2, self.img_shape[0]/2])

                Y = np.matmul( T, (X - self._drone_poses[j]) )

                g = np.matmul( 2*(xn_est - center) , np.matmul(self.G_Y(Y) , T) )
                g[2] = 0
                # g = np.matmul( g, self.X_P(self._times[j]))
                g = np.matmul( g, self.X_P_cos(self._times[j]))
                grad = grad + g

            for j in range(self._gt_poses.shape[0]):
                # ts = self.Ts(self._times[j])
                ts = self.COSs(self._times[j])
                self.new_poses[j,:] = self._gt_poses[j,:] + np.matmul(P, ts).T[0]

            P = P - 1e-5*grad
            self.plot_trajs()

            # print(P)
            # ts = self.Ts(self._times[sel])
            ts = self.COSs(self._times[j])
            X = self._gt_poses[sel,:] + np.matmul(P, ts).ravel()
            # print(X)
            xn_est = self.kin.reproject_single(self._drone_poses[sel], X, self._smooth_eul[sel], self.img_shape)
            xn_est -= np.array([self.img_shape[1]/2, self.img_shape[0]/2])

            first_img = cv.imread(self.img_files[sel])
            rect = (int(xn_est[0]+self.img_shape[1]/2)-1,int(xn_est[1]+self.img_shape[0]/2)-1,2,2)
            cv.rectangle(first_img, rect, (0,255,255), 2)
            rect = (int(self._centers[sel][0]*self._f)-1,int(self._centers[sel][1]*self._f)-1,2,2)
            cv.rectangle(first_img, rect, (0,0,255), 2)
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
parser.add_argument("log_path", help="DJI log path")
args = parser.parse_args()

opt = Optimizer(args.folder_path, args.gt_path, args.log_path)
opt.show()
