import numpy as np
import cv2
import os

def SDMatrixTransformation(img,filterPeople,matrix):

    #InterestPoint is used to record the position of the bounding box we are measuring frmm.
    interestPoint = []
    for i in range(len(filterPeople)):
        interestPoint.append([(filterPeople[i][0]+(filterPeople[i][2]*0.5)), (filterPeople[i][1]+(filterPeople[i][3]*0.99)),1])
        
    #Turn interest point to a numpy array.
    interestPoint = np.array(interestPoint)

    #TransformedPoints is a list that hold the transformed interest points
    transformedPoints = np.zeros_like(interestPoint)
    for i in range(len(transformedPoints)):
        transformedPoints[i] = np.matmul(matrix,interestPoint[i])

    #Normalize the transformed points
    for i in range(len(transformedPoints)):
        transformedPoints[i] = [transformedPoints[i][0] / transformedPoints[i][2],transformedPoints[i][1] / transformedPoints[i][2], 1]

    #Record the people who are social distancing and not.
    socialDistancing = []
    notSocialDistancing = []

    #This for loop plots points that are within the black box resolution and adding people who are and not social distancing.
    for x in range(len(transformedPoints)):
        for y in range(len(transformedPoints)):
            if (transformedPoints[x][0] <= 2600 and transformedPoints[x][1] <= 1300 and transformedPoints[y][0] <= 2600 and transformedPoints[y][1] <= 1300 and x != y):
                distanceBetween = np.sqrt((transformedPoints[x][0] - transformedPoints[y][0]) ** 2 + (transformedPoints[x][1] - transformedPoints[y][1]) ** 2)
                if (distanceBetween < 300):
                    cv2.line(img, (int(transformedPoints[x][0]),int(transformedPoints[x][1])), (int(transformedPoints[y][0]),int(transformedPoints[y][1])), (0, 0, 255), thickness= 10)
                    notSocialDistancing.append(x)
                else:
                    #cv2.line(img, (int(transformedPoints[x][0]),int(transformedPoints[x][1])), (int(transformedPoints[y][0]),int(transformedPoints[y][1])), (0, 255, 0), thickness= 2)
                    socialDistancing.append(x)

    #These for loops draw different coloured circles depending on the pedestrians social distancing status.
    for i in socialDistancing:
        if (transformedPoints[i][0] <= 2600 and transformedPoints[i][1] <= 1200):
            cv2.circle(img,(int(transformedPoints[i][0]),int(transformedPoints[i][1])),50,(0,255,0), 3)

    for i in notSocialDistancing:
        if (transformedPoints[i][0] <= 2600 and transformedPoints[i][1] <= 1200):
            cv2.circle(img,(int(transformedPoints[i][0]),int(transformedPoints[i][1])),50,(0,255,255), 3)



    return img, socialDistancing, notSocialDistancing

