from __future__ import absolute_import
import cv2
import numpy as np
def read_video(vid, interval):
    cap = cv2.VideoCapture(vid)

    frame_count = 0
    frame_list = []
    while True:
        ret, frame = cap.read()
        if ret is False:
            break

        if np.mod(frame_count , interval) == 0:
            frame_list.append(frame)

        frame_count += 1

    frame_array = np.array(frame_list)
    return frame_array
