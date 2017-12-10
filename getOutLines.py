import numpy as np
import cv2

lines = None
points = []
img = None

def draw_circle(event,x,y,flags,param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),5,(255,0,0),-1)
        points.append([x, y])
def getOutLines(frame):
    global lines, points
    global img

    lines = []
    points = []

    img = np.copy(frame)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)

    while(1):
        cv2.imshow('image', img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    if len(points) != 4:
        return getOutLines(frame)
    points[0][0] /= 1.0 * frame.shape[1]
    points[1][0] /= 1.0 * frame.shape[1]
    points[2][0] /= 1.0 * frame.shape[1]
    points[3][0] /= 1.0 * frame.shape[1]

    points[0][1] /= 1.0 * frame.shape[0]
    points[1][1] /= 1.0 * frame.shape[0]
    points[2][1] /= 1.0 * frame.shape[0]
    points[3][1] /= 1.0 * frame.shape[0]

    return [(points[0][0], points[0][1], points[1][0], points[1][1]), (points[2][0], points[2][1], points[3][0], points[3][1])]

