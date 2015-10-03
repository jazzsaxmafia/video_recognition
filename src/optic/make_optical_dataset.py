#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yaml
import ipdb
import os

import cv2

def crop_optic_flow(image, target_height=227, target_width=227):

    height, width = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))

def read_video(vid_file, interval):
    vid = cv2.VideoCapture(vid_file)
    ret, frame1 = vid.read()

    prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    dx_list = []
    dy_list = []

    count = 0

    while True:
        ret, frame2 = vid.read()

        if not ret:
            break

        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        if np.mod(count, interval) == 0:
            flow = cv2.calcOpticalFlowFarneback(prev, next, 0.5, 3, 15, 3, 5, 1.2, 0)
            dx = crop_optic_flow(flow[...,0])
            dy = crop_optic_flow(flow[...,1])

            dx_list.append(dx)
            dy_list.append(dy)

        count += 1
        prev = next

    dx_list = np.array(dx_list)[None,...]
    dy_list = np.array(dy_list)[None,...]

    return np.vstack([dx_list, dy_list]).swapaxes(1,0)
#    dx_mean = np.mean(dx_list, axis=0)
#    dy_mean = np.mean(dy_list, axis=0)

#    return crop_optic_flow(dx_mean), crop_optic_flow(dy_mean)

def make_optic_file(vid_file, data_path, save_path, interval):
    if os.path.exists(os.path.join(save_path, vid_file + '.dx.jpg')):
        print vid_file, " is already processed"
        return
    print "processing ", vid_file
    dx, dy = read_video(os.path.join(data_path,vid_file), interval)

    cv2.imwrite(os.path.join(save_path, vid_file + '.dx.jpg'), dx)
    cv2.imwrite(os.path.join(save_path, vid_file + '.dy.jpg'), dy)

#interval = 10
#data_path = '../data/UCF_video'
#save_path = '../data/UCF_optic'
#video_list = os.listdir(data_path)
#
#map(lambda vid_file: make_optic_file(vid_file, data_path, save_path, interval), video_list)
