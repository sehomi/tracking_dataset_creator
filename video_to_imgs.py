#!/usr/bin/env python

import cv2 as cv
import numpy as np
import os
import shutil

video_path = "vids/drone_pixel_7.avi"
skip_rate = 3

def get_input(question):
    key = -1
    while True:
        frame_copy = frame.copy()
        cv.putText(frame_copy, question + " y or n?" ,(20,20), cv.FONT_HERSHEY_PLAIN, 1, [0,0,255], 2)
        cv.imshow(win_name, frame_copy)
        key = cv.waitKey()

        if key == ord("y"):
            return 1
        elif key == ord("n"):
            return 0


def wrtie_camera_motion(dir):
    f = open(dir+"/camera_motion.tag", "a")
    f.write("{:d}\n".format(get_input("camera motion")))
    f.close()

def wrtie_illum_change(dir):
    f = open(dir+"/illum_change.tag", "a")
    f.write("{:d}\n".format(get_input("illum change")))
    f.close()

def wrtie_occlusion(dir):
    f = open(dir+"/occlusion.tag", "a")
    f.write("{:d}\n".format(get_input("occlusion")))
    f.close()

def wrtie_motion_change(dir):
    f = open(dir+"/motion_change.tag", "a")
    f.write("{:d}\n".format(get_input("motion change")))
    f.close()

def wrtie_size_change(dir):
    f = open(dir+"/size_change.tag", "a")
    f.write("{:d}\n".format(get_input("size change")))
    f.close()

def write_ground_truth(dir):
    rect = cv.selectROI(win_name,frame)
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[0] + rect[2]
    y2 = rect[1]
    x3 = rect[0] + rect[2]
    y3 = rect[1] + rect[3]
    x4 = rect[0]
    y4 = rect[1] + rect[3]

    f = open(dir+"/groundtruth.txt", "a")
    f.write("{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(x1,y1,x2,y2,x3,y3,x4,y4))
    f.close()

win_name = "dataset_creator"
_, file_name = os.path.split(video_path)
video_name, _ = os.path.splitext(file_name)

seq_dir = "custom_dataset/"+video_name
if os.path.exists(seq_dir):
    shutil.rmtree(seq_dir)
os.makedirs(seq_dir)

cap = cv.VideoCapture(video_path)

frame_counter=1
skip_counter = 0
while True:
    ret, frame = cap.read()

    if skip_counter < skip_rate:
        skip_counter += 1
        continue

    skip_counter = 0

    if not ret:
        print("End of video")
        break

    wrtie_camera_motion(seq_dir)
    wrtie_illum_change(seq_dir)
    wrtie_occlusion(seq_dir)
    wrtie_motion_change(seq_dir)
    wrtie_size_change(seq_dir)
    write_ground_truth(seq_dir)

    img_name = seq_dir + "/" + "{:08d}".format(frame_counter) + ".jpg"
    cv.imwrite(img_name, frame)
    frame_counter += 1
