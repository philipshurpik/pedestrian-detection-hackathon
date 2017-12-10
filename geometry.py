import numpy as np
import math
 
def getDistances(axisL, axisR, farDistance, bodies):
	distances = []
 
	farWidth = dist(axisL[2], 0, axisR[2], 0)
	nearWidth =  dist(axisL[0], 0, axisR[0], 0)
	l0 = (farDistance * farWidth) / (nearWidth - farWidth)
 
	for i in bodies:
		botY = getBotOfBox(i)
		distances.append(calcDistance(axisL, axisR, farWidth, farDistance, l0, botY))
 
	return distances
 
 
def dist(x1, y1, x2, y2):
	return math.sqrt((x1 - x2) * (x1 - x2) +  (y1 - y2) * (y1 - y2))
 
def getBotOfBox(box):
	return box[1] + box[3] * 0.5 #box.y + box.h
 
def calcDistance(axisL, axisR, farWidth, farDistance, l0, botY):
	botX_L = getXOnAxis(axisL, botY)
	botX_R = getXOnAxis(axisR, botY)
 
	farCurWidth = dist(botX_L, 0, botX_R, 0)
 
	#dist = farDist * (farWidth / curWidth)
	#return farDistance * (farWidth / (dist(botX_L, 0, botX_R, 0)))
	return l0 + farDistance - l0 * farCurWidth / farWidth
 
def getXOnAxis(axis, y):
	#x = x1 + (x2 - x1) * ((y1 - y) / (y1 - y2))
	return axis[0] + (axis[2] - axis[0]) * ((axis[1] - y) / (axis[1] - axis[3]))


#danger estimation
 
def compute_closest_ped(bBoxArray, point):
    min_bbox = bBoxArray[0]
    min_distance = 1000
    for bBox in bBoxArray:
        distance = math.sqrt(math.pow((bBox[0]-point[0]),2)+math.pow((bBox[1]+bBox[3]*0.5-point[1]),2))
        if distance<min_distance:
            min_distance = distance
            min_bbox = bBox
    return min_bbox

def onRoad(axisL, axisR, body):
    leftX = getXOnAxis(axisL, body[1])
    rightX = getXOnAxis(axisR, body[1])

    return (body[0] >= leftX and body[1] <= rightX)

def estimate(matrix, axisL, axisR, farDistance, bodies):
    #const exDist
    exDist = 3

    ans = np.zeros(len(matrix), (len(matrix[0])))

    if (len(bodies) == 0):
        return ans

    n = len(matrix)
    m = 0
    if (len(matrix) != 0):
        m = len(matrix[0])

    for i in range(n):
        for j in range(m):
            if (matrix[i][j] == 0):
                continue
            x = i / n
            y = j / m
            nearBox = compute_closest_ped(bodies, [x, y])
            distancesToCamera = getDistances(axisL, axisR, farDistance, [[x, y], nearBox])
            distToNear = math.abs(distancesToCamera[0] - distancesToCamera[1])
            if (distToNear <= exDist):
                rate = 1.0 - (distToNear / exDist)
                if (onRoad(axisL, axisR, [nearBox[0], nearBox[1] + nearBox[3] * 0.5])):
                    rate = rate / 2 + 0.5
                else:
                    rate = rate * (3 / 4)

            ans[i][j] = rate

    return ans