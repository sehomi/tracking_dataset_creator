#!/usr/bin/env python

import numpy as np
import os
import shutil
import argparse
import sys
import cv2 as cv
import utils as utl
import matplotlib.pyplot as plt
import numpy.fft as fft
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import rosbag
from cv_bridge import CvBridge


def get_frequency(values, ts):
    ds = []
    for i, t in enumerate(ts):
        if i!=0:
            ds.append(ts[i]-ts[i-1])
    d = np.mean(ds)
    print("sample distance: ", d)

    spectrum = fft.fft(values)
    freq = fft.fftfreq(len(spectrum),d=d)

    idx = abs(freq) < 0.25
    spectrum[idx] = 0

    fig, ax = plt.subplots()
    ax.plot(freq, abs(spectrum), linewidth=2)

    peaks = freq[np.argmax(abs(spectrum))]
    print("dominant frequency: ",abs(peaks))

    plt.show()

    return abs(peaks)

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

def write_cam_stats(dir, pos, eul, t):

    f = open(dir+"/camera_states.txt", "a")
    f.write("{:.3f},{:.2f},{:.2f},{:.2f},{:.4f},{:.4f},{:.4f}\n".format(t,pos[0],pos[1],pos[2],\
                                                                        eul[0],eul[1],eul[2]))
    f.close()

def write_test_config(target_dir):

    open(target_dir + "/test_config.txt", "w").close()


parser = argparse.ArgumentParser()
parser.add_argument("bag_file", help="Path to the bag file.")
args = parser.parse_args()

bag_file = args.bag_file
file_name = os.path.splitext(os.path.basename(bag_file))[0]

bag = rosbag.Bag(bag_file)
bridge = CvBridge()

win_name = "dataset_creator"
cv.namedWindow(win_name)
cv.setMouseCallback(win_name, get_input_btn)

tracking_challenges = {"camera motion":0, "illum change":0, "occlusion":0, \
                       "motion change":0, "size change":0}
btn_bar_corner = [0,0]

pitches = []
ts = []
t0 = 0
for topic, msg, t in bag.read_messages(topics=['/tello/odom']):

    if topic == '/tello/odom':
        ori = msg.pose.pose.orientation
        roll, pitch, yaw = utl.quaternion_to_euler_angle(ori.w, ori.x, ori.y, ori.z)
        pitches.append(pitch)
        if len(ts)==0:
            t0 = t.to_sec()
        ts.append(t.to_sec()-t0)

freq = get_frequency(pitches, ts)

fig, ax = plt.subplots()
ax.plot(ts, pitches, linewidth=2)
plt.show()

seq_dir = "dataset/custom_dataset1/cup_{:.1f}HZ".format(freq)
if os.path.exists(seq_dir):
    shutil.rmtree(seq_dir)
os.makedirs(seq_dir)

write_test_config(seq_dir)
static_cam_pos = [-1, 0, 0.35]

tracker = cv.TrackerCSRT_create()
# tracker = cv.Tracker_create("csrt")


frame_counter=1
start_img_num=None
odom_eul=None
t0 = 0
for topic, msg, t in bag.read_messages(topics=['/tello/odom','/tello/camera/image_raw']):

    if t0==0:
        t0 = t.to_sec()

    if topic == '/tello/odom':
        ori = msg.pose.pose.orientation
        roll, pitch, yaw = utl.quaternion_to_euler_angle(ori.w, ori.x, ori.y, ori.z)
        odom_eul = [roll, pitch, yaw]


    if topic == '/tello/camera/image_raw':

        if odom_eul is None:
            continue

        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        frame = cv.resize(frame, (640,480))

        if t.to_sec()-t0 < 0.5:
            continue
        # cv.imshow(win_name, frame)
        # k = cv.waitKey()
        # continue

        if frame_counter==1:
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
        write_cam_stats(seq_dir, static_cam_pos, odom_eul, t.to_sec())

        # TODO: write camera and target positions to log file

        img_name = seq_dir + "/" + "{:08d}".format(frame_counter) + ".jpg"
        cv.imwrite(img_name, frame)
        frame_counter += 1
