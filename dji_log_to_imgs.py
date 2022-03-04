#!/usr/bin/env python

import cv2 as cv
import numpy as np
import os
import shutil

log_path = "dji_logs/log_2022_02_15_16_44_45"
skip_rate = 0

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

def write_cam_stats(dir, num):
    global data_lines

    line = data_lines[num-1]
    data = line.split(',')
    img_num = int(data[0][5:].split('.')[0])

    if img_num != num:
        return False

    f = open(dir+"/camera_states.txt", "a")
    f.write("{:s},{:s},{:s}".format(data[5],data[6],data[7]))
    f.close()

    return True


win_name = "dataset_creator"
_, folder_name = os.path.split(log_path)

log_data_file = open(log_path + '/log.txt', 'r')
data_lines = log_data_file.readlines()

number_of_imgs = len(os.listdir(log_path)) - 1
if number_of_imgs == len(data_lines):
    print("Number of images and log lines match. proceeding ...")
else:
    print("Number of images and log lines DO NOT match ({:d} images and {:d} lines). exiting ...".format(number_of_imgs, len(data_lines)))
    exit()

seq_dir = "custom_dataset1/"+folder_name
if os.path.exists(seq_dir):
    shutil.rmtree(seq_dir)
os.makedirs(seq_dir)

counter=1
frame_counter=1
skip_counter = 0
while True:
    try:
        frame = cv.imread(log_path + "/image{:d}.jpg".format(counter))

        counter += 1
        if get_input("skip?"):
            continue
    except:
        if counter >= number_of_imgs:
            print("End of images")
        else:
            print("Error reading " + log_path + "/image{:d}.jpg".format(counter))

        break

    if skip_counter < skip_rate:
        skip_counter += 1
        continue

    skip_counter = 0

    if not write_cam_stats(seq_dir, counter):
        print("Error finding log for " + log_path + "/image{:d}.jpg".format(counter-1))
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
