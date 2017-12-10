import os
import argparse
import itertools 

import cv2
import numpy as np
import math

from geometry import getDistances
from getOutLines import getOutLines

def getConstSeparateLines():
    left = np.array([0, 0.9, 0.4, 0.6])
    right = np.array([1.0, 0.9, 0.6, 0.6])

    return [left, right]

def getSeparateLines(frame=None):
    # return getConstSeparateLines()
    return getOutLines(frame)

    # debug 
    frame = cv2.imread('car.jpg')
    # frame = np.zeros((500, 500), dtype=np.float32)
    for line in [left, right]:
        line[0] *= frame.shape[1]
        line[1] *= frame.shape[0]
        line[2] *= frame.shape[1]
        line[3] *= frame.shape[0]
        line = [int(coord) for coord in line]
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (255, 255, 255), 2)
    cv2.imshow('debug_lines', frame)
    cv2.waitKey(0)
    #end of debug

def run_pedestrian_detector(frame):
    boxes = [[0.5, 0.5, 0.3, 0.3]]
    return boxes

def run_segmentation(frame):
    road_mask = np.zeros(frame.shape[:2],dtype=np.float32)
    return road_mask

parser = argparse.ArgumentParser(description='pedestrian detection')
parser.add_argument('--video_dir', default='test_videos/pedestrian-1.mp4', type=str, help='video path')
parser.add_argument('--camera_dist', default=20, type=float, help='camera dist')
args = parser.parse_args()

videoCap = cv2.VideoCapture(0) if args.video_dir == '' else cv2.VideoCapture(args.video_dir)
lines = None

while(True):
    ret, frame = videoCap.read()
    if frame is None:
        break

    r, g, b = cv2.split(frame)
    frame2show = cv2.merge([r, g, b])


    if lines == None:
        lines = getSeparateLines(frame2show)

    #draw line
    for nline in lines:
        line = np.copy(nline)
        line[0] *= frame.shape[1]
        line[1] *= frame.shape[0]
        line[2] *= frame.shape[1]
        line[3] *= frame.shape[0]
        line = [int(x) for x in line]

        cv2.line(frame2show, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 2)

    boxes = run_pedestrian_detector(frame)
    # info = [(box, estimate_distance(box)) for box in boxes]
    distances = getDistances(lines[0], lines[1], args.camera_dist, boxes)
    info = zip(boxes, distances)
    road_segment = run_segmentation(frame)

    def draw_info(frame, info):
        def estimate_color(box, dist):
            if dist > 30:
                dist = 30
            elif dist < 0:
                dist = 0.0
            g = 255 * (dist / 30.0)
            r = 255 * ((30 - dist) / 30.0)
            return (0, g, r)

        for box, distance in info:
            x, y, w, h = box

            x *= frame.shape[1]
            y *= frame.shape[0]
            w *= frame.shape[1]
            h *= frame.shape[0]

            x -= w * 0.5
            y -= h * 0.5
            x, y, w, h = int(x), int(y), int(w), int(h)

            color = estimate_color(box, distance)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)
            cv2.putText(frame, str(distance), (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    
    draw_info(frame2show, info)
    cv2.imshow('frame', frame2show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
videoCap.release()
cv2.destroyAllWindows()