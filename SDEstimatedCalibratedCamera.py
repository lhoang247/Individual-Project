import cv2
import numpy as np
from CameraMatrix import returnCameraParameters
import matplotlib.pyplot as plt
def SDCalibratedCamera(img,filterPeople):
    rm, tm, cm, _ = returnCameraParameters(6)

    pList = []

    realWorld = np.array([[0,0],[1,0],[1,1],[0,1]])
    imagePoints = np.array([[520, 478],[860, 444],[1178, 590],[715, 658]])

    h, status = cv2.findHomography(imagePoints, realWorld)

    xPlot = []
    yPlot = []


    for i in range(len(filterPeople)):

        u = int(filterPeople[i][0] + filterPeople[i][2]/2)
        v = int(filterPeople[i][1] + filterPeople[i][3]*99)

        coordinates = np.array([[u],[v],[1]])


        test = np.matmul(h,coordinates)

        pList.append(test / test[2])
        xPlot.append(test[0]/test[2])
        yPlot.append(test[1]/test[2])
    
    plt.plot(xPlot,yPlot, 'o', color = 'black')
    plt.show()
    #lowestX = 99999
    #highestX = 99999 
    #lowestY = 99999
    #highestY = 99999
    #for j in range(len(pList)):
    #    if ((pList[j][0] < lowestX and abs(pList[j][0]) < 10)  or lowestX == 99999):
    #        lowestX = pList[j][0]
    #    if ((pList[j][0] > highestX and abs(pList[j][0]) <10) or highestX == 99999):
    #        highestX = pList[j][0]
    #    if ((pList[j][1] < lowestY and abs(pList[j][1]) <10) or lowestY == 99999):
    #        lowestY = pList[j][1]
    #    if ((pList[j][1] > highestY and abs(pList[j][1]) < 10) or highestY == 99999):
    #        highestY = pList[j][1]
#
    #rangeX = highestX - lowestX
    #rangeY = highestY - lowestY
    #for i in range(len(pList)):
    #    #pList[i][0] = (((pList[i][0] - lowestX)) / rangeX) * 300
    #    #pList[i][1] = (((pList[i][1] - lowestY)) / rangeY) * 500
    #    #print("(", (((pList[i][0] - lowestX)) / rangeX) * 300 , ", ",(((pList[i][1] - lowestY)) / rangeY) * 500 , ")")
    #    cv2.circle(img,(int((pList[i][0]- 0.09508975/2)*3000+3350),int((pList[i][1] - 0.14209866/2)*5000-13100)),10,(255,0,0), -1)

    for x in range(len(filterPeople)):
        cv2.rectangle(img, (filterPeople[x][0],filterPeople[x][1]), (filterPeople[x][0]+filterPeople[x][2], filterPeople[x][1]+filterPeople[x][3]), (255,255,0), 2)
        for y in range(len(filterPeople)):
            if (x != y):
                distance = np.sqrt(abs(pList[x][0]-pList[y][0])**2 + abs(pList[x][1]-pList[y][1])**2)
                if (distance < 0.015):
                    cv2.line(img, (int(filterPeople[x][0]+(filterPeople[x][2]*0.5)), int(filterPeople[x][1]+filterPeople[x][3]*0.99)), (int(filterPeople[y][0]+(filterPeople[y][2]*0.5)), int(filterPeople[y][1]+filterPeople[y][3]*0.99)), (0, 0, 255), thickness= 2)

    return img, blackBox