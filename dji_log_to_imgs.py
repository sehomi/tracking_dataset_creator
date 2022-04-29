#!/usr/bin/env python

import cv2 as cv
import numpy as np
import os
import shutil


def get_input_key(question, rect):

    cv.setMouseCallback(win_name, get_input_btn)
    key = -1
    while True:
        frame_copy = frame.copy()
        frame_copy = create_buttons(frame_copy)
        cv.rectangle(frame_copy, rect, (0,255,255), 2)
        cv.putText(frame_copy, question + " y or n?" ,(20,20), cv.FONT_HERSHEY_PLAIN, 1, [0,0,255], 2)
        cv.imshow(win_name, frame_copy)
        key = cv.waitKey(33)

        if key == ord("y"):
            return 1
        elif key == ord("n"):
            return 0

def get_input_btn(event, x, y, flags, param):
    global btn_bar_corner

    X = x
    Y = y - btn_bar_corner[1]

    if event == cv.EVENT_LBUTTONDOWN:
        if X > 0 and X <= 160 and Y > 0 and Y <= 40:
            tracking_challenges["camera motion"] = 1-tracking_challenges["camera motion"]
        elif X > 160 and X <= 300 and Y > 0 and Y <= 40:
            tracking_challenges["illum change"] = 1-tracking_challenges["illum change"]
        elif X > 300 and Y > 0 and Y <= 40:
            tracking_challenges["occlusion"] = 1-tracking_challenges["occlusion"]
        elif X > 0 and X <= 160 and Y > 40:
            tracking_challenges["motion change"] = 1-tracking_challenges["motion change"]
        elif X > 160 and X <= 300 and Y > 40:
            tracking_challenges["size change"] = 1-tracking_challenges["size change"]

def create_buttons(image):
    global tracking_challenges, btn_bar_corner

    w = image.shape[1]
    h = 80

    buttins_bar = np.ones((h,w,3), dtype=np.uint8)*255

    cv.putText(buttins_bar, "camera motion" ,(20,20), cv.FONT_HERSHEY_PLAIN, 1,\
              [0,0,255*tracking_challenges["camera motion"]], 2)
    cv.putText(buttins_bar, "illum change" ,(180,20), cv.FONT_HERSHEY_PLAIN, 1,\
              [0,0,255*tracking_challenges["illum change"]], 2)
    cv.putText(buttins_bar, "occlusion" ,(320,20), cv.FONT_HERSHEY_PLAIN, 1, \
              [0,0,255*tracking_challenges["occlusion"]], 2)
    cv.putText(buttins_bar, "motion change" ,(20,50), cv.FONT_HERSHEY_PLAIN, 1,\
              [0,0,255*tracking_challenges["motion change"]], 2)
    cv.putText(buttins_bar, "size change" ,(180,50), cv.FONT_HERSHEY_PLAIN, 1, \
              [0,0,255*tracking_challenges["size change"]], 2)

    res = np.concatenate((image, buttins_bar), axis=0)

    btn_bar_corner = [0,frame.shape[0]]

    return res


def wrtie_camera_motion(dir, val):
    f = open(dir+"/camera_motion.tag", "a")
    # f.write("{:d}\n".format(get_input_key("camera motion")))
    f.write("{:d}\n".format(val))
    f.close()

def wrtie_illum_change(dir, val):
    f = open(dir+"/illum_change.tag", "a")
    # f.write("{:d}\n".format(get_input_key("illum change")))
    f.write("{:d}\n".format(val))
    f.close()

def wrtie_occlusion(dir, val):
    f = open(dir+"/occlusion.tag", "a")
    # f.write("{:d}\n".format(get_input_key("occlusion")))
    f.write("{:d}\n".format(val))
    f.close()

def wrtie_motion_change(dir, val):
    f = open(dir+"/motion_change.tag", "a")
    # f.write("{:d}\n".format(get_input_key("motion change")))
    f.write("{:d}\n".format(val))
    f.close()

def wrtie_size_change(dir, val):
    f = open(dir+"/size_change.tag", "a")
    # f.write("{:d}\n".format(get_input_key("size change")))
    f.write("{:d}\n".format(val))
    f.close()

def write_ground_truth_box(dir,rect=None):
    ret = False
    if rect is None:
        ret = True
        rect = cv.selectROI(win_name,frame)
        if rect[2] <= 1:
            rect[2] = (rect[0], rect[1], 2, rect[3])
        if rect[3] <= 1:
            rect[3] = (rect[0], rect[1], rect[2], 2)

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

    if ret:
        return rect

def write_cam_stats(dir, num, start):
    global data_lines

    line = data_lines[num-start]
    data = line.split(',')
    img_num = int(data[0][5:].split('.')[0])

    if img_num != num-1:
        return False

    f = open(dir+"/camera_states.txt", "a")
    f.write("{:s},{:s},{:s}".format(data[5],data[6],data[7]))
    f.close()

    return True



log_path = "raw_data/dji_logs/park_mavic_1"
skip_rate = 0

win_name = "dataset_creator"
cv.namedWindow(win_name)
cv.setMouseCallback(win_name, get_input_btn)

_, folder_name = os.path.split(log_path)

log_data_file = open(log_path + '/log.txt', 'r')
data_lines = log_data_file.readlines()

tracking_challenges = {"camera motion":0, "illum change":0, "occlusion":0, \
                       "motion change":0, "size change":0}
btn_bar_corner = [0,0]

files = [f for f in os.listdir(log_path) if os.path.isfile(os.path.join(log_path, f))]
img_files = []
for file in files:
    if file.endswith(".jpg"):
        img_files.append(file)
number_of_imgs = len(img_files)
if number_of_imgs == len(data_lines):
    print("Number of images and log lines match. proceeding ...")
else:
    print("Number of images and log lines DO NOT match ({:d} images and {:d} lines). exiting ...".format(number_of_imgs, len(data_lines)))
    exit()

seq_dir = "dataset/custom_dataset1/"+folder_name
if os.path.exists(seq_dir):
    shutil.rmtree(seq_dir)
os.makedirs(seq_dir)

tracker = cv.TrackerCSRT_create()

counter=1
frame_counter=1
start_img_num=None
skip_counter = 0
while True:
    try:
        frame = cv.imread(log_path + "/image{:d}.jpg".format(counter))

        counter += 1

        if frame is None:
            continue
        elif frame is not None and start_img_num is None:
            start_img_num=counter

        # if get_input_key("skip?"):
        #     continue
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

    if not write_cam_stats(seq_dir, counter, start_img_num):
        print("Error finding log for " + log_path + "/image{:d}.jpg".format(counter-1))
        break


    if counter==start_img_num:
        roi = write_ground_truth_box(seq_dir)
        tracker.init(frame, roi)
    else:
        _, res = tracker.update(frame)
        res = tuple( [int(val) for val in res] )
        if get_input_key("rect is good.", res):
            write_ground_truth_box(seq_dir, res)
        else:
            roi = write_ground_truth_box(seq_dir)
            tracker = cv.TrackerCSRT_create()
            tracker.init(frame, roi)

    wrtie_camera_motion(seq_dir, tracking_challenges["camera motion"])
    wrtie_illum_change(seq_dir, tracking_challenges["illum change"])
    wrtie_occlusion(seq_dir, tracking_challenges["occlusion"])
    wrtie_motion_change(seq_dir, tracking_challenges["motion change"])
    wrtie_size_change(seq_dir, tracking_challenges["size change"])

    # TODO: write camera and target positions to log file

    img_name = seq_dir + "/" + "{:08d}".format(frame_counter) + ".jpg"
    cv.imwrite(img_name, frame)
    frame_counter += 1
