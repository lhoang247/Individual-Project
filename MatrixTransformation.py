import numpy as np
import cv2
import os

def SDMatrixTransformation(img,filterPeople,matrix):
    interestPoint = []
    for i in range(len(filterPeople)):
        interestPoint.append([(filterPeople[i][0]+(filterPeople[i][2]*0.5)), (filterPeople[i][1]+(filterPeople[i][3]*0.99)),1])
        

    interestPoint = np.array(interestPoint)
    transformedPoints = np.zeros_like(interestPoint)
    for i in range(len(transformedPoints)):
        transformedPoints[i] = np.matmul(matrix,interestPoint[i])


    for i in range(len(transformedPoints)):
        transformedPoints[i] = [transformedPoints[i][0] / transformedPoints[i][2],transformedPoints[i][1] / transformedPoints[i][2], 1]

    socialDistancing = []
    notSocialDistancing = []

    for x in range(len(transformedPoints)):
        for y in range(len(transformedPoints)):
            if (transformedPoints[x][0] <= 2600 and transformedPoints[x][1] <= 1200 and transformedPoints[y][0] <= 2600 and transformedPoints[y][1] <= 1200 and x != y):
                distanceBetween = np.sqrt((transformedPoints[x][0] - transformedPoints[y][0]) ** 2 + (transformedPoints[x][1] - transformedPoints[y][1]) ** 2)
                if (distanceBetween < 300):
                    cv2.line(img, (int(transformedPoints[x][0]),int(transformedPoints[x][1])), (int(transformedPoints[y][0]),int(transformedPoints[y][1])), (0, 0, 255), thickness= 10)
                    notSocialDistancing.append(x)
                else:
                    #cv2.line(img, (int(transformedPoints[x][0]),int(transformedPoints[x][1])), (int(transformedPoints[y][0]),int(transformedPoints[y][1])), (0, 255, 0), thickness= 2)
                    socialDistancing.append(x)

    for i in socialDistancing:
        if (transformedPoints[i][0] <= 2600 and transformedPoints[i][1] <= 1200):
            cv2.circle(img,(int(transformedPoints[i][0]),int(transformedPoints[i][1])),50,(0,255,0), 3)

    for i in notSocialDistancing:
        if (transformedPoints[i][0] <= 2600 and transformedPoints[i][1] <= 1200):
            cv2.circle(img,(int(transformedPoints[i][0]),int(transformedPoints[i][1])),50,(0,255,255), 3)



    return img, socialDistancing, notSocialDistancing

