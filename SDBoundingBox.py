import numpy as np
import cv2
import os

def SDBoundingBox(img,filterPeople):
    for i in range(len(filterPeople)):
            SD = True
            for j in range(len(filterPeople)):
                if (i != j):
                    pixelDistance = np.sqrt(abs(float(filterPeople[i][0]+(filterPeople[i][2]*0.5)) - float(filterPeople[j][0]+(filterPeople[j][2]*0.5)))**2 + abs(float(filterPeople[i][1]+(filterPeople[i][3]*0.99))*2 - float(filterPeople[j][1]+(filterPeople[j][3]*0.99))*2)**2)
                    middleDistance = (filterPeople[i][1]+filterPeople[i][2]+filterPeople[j][1]+filterPeople[j][2]) / 2
                    if (pixelDistance < 400 - (800/middleDistance)*20):
                        cv2.line(img, (int(filterPeople[i][0]+(filterPeople[i][2]*0.5)), int((filterPeople[i][1]+filterPeople[i][3]*0.99))), (int(filterPeople[j][0]+(filterPeople[j][2]*0.5)), int((filterPeople[j][1]+filterPeople[j][3]*0.99))), (255, 255, 0), thickness= 2)
                        SD = False
                    #elif ( pixelDistance < 600):
                        #cv2.line(img, (int(filterPeople[i][0]+(filterPeople[i][2]*0.5)), int((filterPeople[i][1]+filterPeople[i][3]*0.99))), (int(filterPeople[j][0]+(filterPeople[j][2]*0.5)), int((filterPeople[j][1]+filterPeople[j][3]*0.99))), (0, 255, 0), thickness= 2)
            if (SD):
                cv2.rectangle(img, (filterPeople[i][0],filterPeople[i][1]), (filterPeople[i][0]+filterPeople[i][2], filterPeople[i][1]+filterPeople[i][3]), (0,255,0), 2)
            else:
                cv2.rectangle(img, (filterPeople[i][0],filterPeople[i][1]), (filterPeople[i][0]+filterPeople[i][2], filterPeople[i][1]+filterPeople[i][3]), (0,0,255), 2)
    
    return img