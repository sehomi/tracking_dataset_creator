from os.path import join
import numpy as np
import pandas as pd
import argparse
import os
import scipy.io
import cv2 as cv
from mmcv import Config

import torch

import tensorflow as tf
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

import CSENDistance.csen_regressor.model as  mdl

from mono.model.registry import MONO
from mono.model.mono_baseline.layers import disp_to_depth
from mono.datasets.utils import readlines
from mono.datasets.kitti_dataset import KITTIRAWDataset

class RangeEstimator:

    def __init__(self, img_size, method='direct_fcn', direct_mode='normal', \
                 csen_pkg_path='', gcn_pkg_path='', margin=20):

        assert method in ['proportionality', 'direct_fcn', 'direct_csen', 'direct_gcn']
        assert direct_mode in ['normal', 'oblique']
        assert len(img_size) >= 2

        self.method = method
        self.direct_mode = direct_mode
       
        self.img_w = img_size[0]
        self.img_h = img_size[1]
        self.margin = margin
        self.last_z = None
        self.noise_mag = 0.2

        if method == 'direct_fcn':
            self.loadFCNModel()

        elif method == 'direct_csen':
            self.loadCSENModel(csen_pkg_path)

        elif method == 'direct_gcn':
            self.loadGCNModel(gcn_pkg_path)
            self.MIN_DEPTH=1e-3
            self.MAX_DEPTH=80
            self.SCALE = 36


    def transform(self, cv2_img, height=320, width=1024):
      im_tensor = torch.from_numpy(cv2_img.astype(np.float32)).cuda().unsqueeze(0)
      im_tensor = im_tensor.permute(0, 3, 1, 2).contiguous()
      im_tensor = torch.nn.functional.interpolate(im_tensor, [height, width],mode='bilinear', align_corners=False)
      im_tensor /= 255
      return im_tensor

    def loadGCNModel(self, pkg_path):

      cfg_path = pkg_path + '/config/cfg_kitti_fm.py'
      model_path = pkg_path + '/weights/epoch_20.pth'

      cfg = Config.fromfile(cfg_path)
      cfg['model']['depth_pretrained_path'] = None
      cfg['model']['pose_pretrained_path'] = None
      cfg['model']['extractor_pretrained_path'] = None
      self.model = MONO.module_dict[cfg.model['name']](cfg.model)
      checkpoint = torch.load(model_path)
      self.model.load_state_dict(checkpoint['state_dict'], strict=True)
      self.model.cuda()
      self.model.eval()

    def loadFCNModel(self):

        model_name = 'model@1535470106'
        model_path = os.path.dirname(__file__) + '/KITTI-distance-estimation/distance-estimator/generated_files'
        data_path = os.path.dirname(__file__) + '/KITTI-distance-estimation/distance-estimator/data'


        # load json and create model
        json_file = open('{}/{}.json'.format(model_path, model_name), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json( loaded_model_json )

        # load weights into new model
        loaded_model.load_weights('{}/{}.h5'.format(model_path, model_name))

        # evaluate loaded model on test data
        loaded_model.compile(loss='mean_squared_error', optimizer='adam')
        self.model = loaded_model

        # get data
        df_test = pd.read_csv('{}/test.csv'.format(data_path))
        X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
        y_test = df_test[['zloc']].values

        # standardized data
        # self.scalar1 = StandardScaler()
        # X_test = self.scalar1.fit_transform(X_test)
        # self.scalar2 = StandardScaler()
        # y_test = self.scalar2.fit_transform(y_test)
        self.y_mean = np.mean(y_test, axis=0)
        self.y_std = np.std(y_test, axis=0)

    def formScaler(self):

        x_train = self.data['x_train'].astype('float32')
        x_dic = self.data['x_dic'].astype('float32')

        m =  x_train.shape[1]
        n =  x_train.shape[2]

        x_dic = np.reshape(x_dic, [len(x_dic), m * n])
        x_train = np.reshape(x_train, [len(x_train), m * n])
        
        scaler = StandardScaler().fit(np.concatenate((x_dic, x_train), axis = 0))

        return scaler

    def preprocess(self, imgDir, bbox, scaler):
        function_name = 'tf.keras.applications.VGG19'
        model = eval(function_name + "(include_top=False, weights='imagenet', input_shape=(64, 64, 3), pooling='max')")

        im = cv.imread(imgDir)

        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[4])
        y2 = int(bbox[5])

        Object = cv.resize(im[y1:y2, x1:x2, :], (64, 64))
        Object = np.expand_dims(cv.cvtColor(Object, cv.COLOR_BGR2RGB), axis=0)

        function_name = 'tf.keras.applications.' + ('VGG19')[:8].lower() + '.preprocess_input'
        Object = eval(function_name + '(Object)')
    
        f = model.predict(Object, verbose=0)

        f = np.transpose(f).astype(np.double)

        phi = self.data['phi'].astype('float32')
        Proj_M = self.data['Proj_M'].astype('float32')

        Y2 = np.matmul(phi, f)
        Y2 = Y2 / np.linalg.norm(Y2)

        prox_Y2 = np.matmul(Proj_M, Y2)

        prox_Y2 = scaler.transform(prox_Y2.T).T

        x_test = np.reshape(prox_Y2, (1, 15, 80))

        x_test = np.expand_dims(x_test, axis=-1)

        return x_test

    def loadCSENModel(self, pkg_path=''):

        modelType = 'CSEN'
        feature_type = 'VGG19'
        weights = True

        MR = '0.5'

        weightsDir = pkg_path + '/weights/' + modelType + '/'

        self.modelFold = mdl.model()
            
        weightPath = weightsDir + feature_type + '_' + MR + '_' + str(1) + '.h5'

        # Testing and performance evaluations.
        self.modelFold.load_weights(weightPath)

        self.data = scipy.io.loadmat(pkg_path + '/CSENdata-2D/VGG19_mr_0.5_run1.mat')

        self.scaler = self.formScaler()

    def isDistantFromBoundary(self, rect):
        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[0] + rect[2]
        y2 = rect[1] + rect[3]

        return x1>self.margin and y1>self.margin and x2<self.img_w-self.margin and y2<self.img_h-self.margin

    def scale_vector(self, v, z):

        if v is None:
            return None

        ## scale a unit vector v based on the fact that third component should be
        ## equal to z
        max_dist = 50
        if v[2] > 0:
            factor = np.abs(z) / np.abs(v[2])
            if np.linalg.norm(factor*v) < max_dist:
                return factor*v
            else:
                return max_dist*v
        elif v[2] <= 0:
            return max_dist*v

    def findRange(self, rect, image):
        if self.method == 'direct_fcn':
            x1 = np.array([[rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]])
            # x2 = self.scalar1.fit_transform(x1)
            x2 = np.zeros((1,4), dtype=np.float32)
            x2[0,0::2] = x1[0,0::2].astype(np.float32)/self.img_w - 0.5
            x2[0,1::2] = x1[0,1::2].astype(np.float32)/self.img_h - 0.5

            y_pred1 = self.model.predict(x2, verbose = 0)
            y_pred2 = y_pred1*self.y_std + self.y_mean
            # y_pred2 = self.scalar2.inverse_transform(y_pred1)
            # print(x1, x2, y_pred1, y_pred2)

            return y_pred2[0][0]

        elif self.method == 'direct_csen':

            x1 = int(rect[0])
            y1 = int(rect[1])
            x2 = int(rect[0]+rect[2])
            y2 = int(rect[1]+rect[3])

            fts = self.preprocess(image, [x1,y1,x2,y1,x2,y2,x1,y2], self.scaler)
            y_pred = self.modelFold.model.predict(fts, verbose = 0)

            return y_pred[0][0]

        elif self.method == 'direct_gcn':

            img = cv.imread(image)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            original_height, original_width = img.shape[:2]
            im_tensor = self.transform(img)

            with torch.no_grad():
              input = {}
              input['color_aug', 0, 0] = im_tensor
              outputs = self.model(input)

            disp = outputs[("disp", 0, 0)]
            disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)
            min_disp = 1/self.MAX_DEPTH
            max_disp = 1/self.MIN_DEPTH
            depth = 1/(disp_resized.squeeze().cpu().numpy()*max_disp + min_disp) * self.SCALE

            x1 = int(rect[0])
            y1 = int(rect[1])
            x2 = int(rect[0]+rect[2])
            y2 = int(rect[1]+rect[3])

            return np.mean(depth[y1:y2, x1:x2])
            # return depth, disp_resized.squeeze().cpu().numpy()

    def findPos(self, image, rect, direction, z, cls='person'):
        assert cls == 'person'

        pos = None
        if 'direct' in self.method:
            if self.isDistantFromBoundary(rect):
                rng = self.findRange(rect, image)
                pos = rng*direction

                self.last_z = np.abs(pos[2])
            else:
                if self.last_z is not None:
                    noise = (np.random.randn()*self.noise_mag) - self.noise_mag/2
                    pos = self.scale_vector(direction, self.last_z + noise)
                else:
                    pos = self.scale_vector(direction, z)

        elif self.method == 'proportionality':
            noise = (np.random.randn()*self.noise_mag) - self.noise_mag/2
            pos = self.scale_vector(direction, z + noise)

        return pos


if __name__ == "__main__":

  img_hor = cv.imread('/content/tracking_dataset_creator/GCNDepth/3_.jpeg')
  img_tbl = cv.imread('/content/tracking_dataset_creator/dataset/VIOT/cup_0.5HZ/00000005.jpg')
  img_obl = cv.imread('/content/tracking_dataset_creator/dataset/VIOT/park_mavic_1/00000035.jpg')
  img_obl1 = cv.imread('/content/tracking_dataset_creator/dataset/VIOT/park_mavic_6/00000177.jpg')

  _re = RangeEstimator([ img_shape[1], img_shape[0]], method='direct_fcn')
  _re_level = RangeEstimator([ img_shape[1], img_shape[0]], method='proportionality')
  _re_csen = RangeEstimator([ img_shape[1], img_shape[0]], method='direct_csen', csen_pkg_path=csen_pkg_path)
  _re_gcn = RangeEstimator([ img_shape[1], img_shape[0]], method='direct_gcn', gcn_pkg_path=gcn_pkg_path)

  print(re.findRange([100,100,100,100], img))
  # depth, disp_resized = re.findRange(None, img)
  # vmax = np.percentile(disp_resized, 95)
  # import matplotlib.pyplot as plt
  # plt.imsave('/content/tracking_dataset_creator/GCNDepth/out_re.png', disp_resized, cmap='magma', vmax=vmax)