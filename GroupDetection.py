import cv2
import numpy as np
import matplotlib.pyplot as plt

def GroupDetection(img, filterPeople, previousFrame, currentFrame, peopleMap):
    
    for x in range(len(filterPeople)):
        cv2.rectangle(img, (filterPeople[x][0],filterPeople[x][1]), (filterPeople[x][0]+filterPeople[x][2], filterPeople[x][1]+filterPeople[x][3]), (255,0,0), 2)

    if not peopleMap:
        for i in range(len(currentFrame)):
            peopleMap.append([i,currentFrame[i][0],currentFrame[i][1]])


    for i in range(len(currentFrame)):
        for j in range(len(peopleMap)):
            if (abs(currentFrame[i][0] - peopleMap[j][1]) < 20 and abs(currentFrame[i][1] - peopleMap[j][2]) < 20):
                peopleMap[j][1] = currentFrame[i][0]
                peopleMap[j][2] = currentFrame[i][1]


    font = cv2.FONT_HERSHEY_PLAIN


    for i in range(len(peopleMap)):
        cv2.putText(img,str(peopleMap[i][0]), (peopleMap[i][1],peopleMap[i][2] - 50), font , 2, (0,127,255), 2)
    
    return img, peopleMap