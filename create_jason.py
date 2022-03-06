#!/usr/bin/env python

import os
import csv
import json

items = os.listdir('dataset/custom_dataset1')
jason_data = {}
for item in items:
    seq = 'dataset/custom_dataset1/'+item
    if os.path.isfile(seq):
        continue

    img_names = []
    for file in sorted(os.listdir(seq)):
        if file.endswith(".jpg"):
            img_names.append(os.path.join(item, file))

    f = open(seq+"/groundtruth.txt", "r")
    lines = []
    for line in f.readlines():
        line = [float(val) for val in line.split(',')]
        lines.append(line)
    # print(lines[0])

    f1 = open(seq+"/camera_motion.tag", "r")
    cam_mot = []
    for line in f1.readlines():
        cam_mot.append(int(line))
    # print(cam_mot)

    f2 = open(seq+"/illum_change.tag", "r")
    illum_change = []
    for line in f2.readlines():
        illum_change.append(int(line))
    # print(illum_change)

    f3 = open(seq+"/motion_change.tag", "r")
    mot_change = []
    for line in f3.readlines():
        mot_change.append(int(line))
    # print(mot_change)

    f4 = open(seq+"/size_change.tag", "r")
    size_change = []
    for line in f4.readlines():
        size_change.append(int(line))
    # print(size_change)

    f5 = open(seq+"/occlusion.tag", "r")
    occ = []
    for line in f5.readlines():
        occ.append(int(line))
    # print(occ)

    seq_jason = {"video_dir": item, "init_rect": lines[0], "img_names": img_names, \
                 "gt_rect": lines, "camera_motion": cam_mot, "illum_change": illum_change, \
                 "motion_change": mot_change, "size_change": size_change, "occlusion": occ}
    jason_data[item] = seq_jason
    print(item)


with open('dataset/custom_dataset1/custom_dataset1.json', 'w') as outfile:
    json.dump(jason_data, outfile)
