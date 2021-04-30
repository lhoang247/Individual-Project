#Importing libraries

import numpy as np
import cv2
import os
import timeit
import random
import matplotlib.pyplot as plt
from skimage import util, data
from SDBoundingBox import SDBoundingBox
from MatrixTransformation import SDMatrixTransformation
from CameraMatrix import returnCameraParameters
from SDCalibratedCamera import SDCalibratedCamera

#Setting up directories

dir = 'D:\year3\IN3007-IndividualProject\YOLOv3'
pathYoloWeights = dir + '\yolov3.weights'
pathYoloCfg =  dir + '\yolov3.cfg'
pathCoco =  dir + '\coco.names'

#Reading the weights and configs of the neural network architecture

net = cv2.dnn.readNet(pathYoloWeights, pathYoloCfg)

#Implementing GPU compatibility

#Reading what the COCO dataset can identify

classes = []

with open(pathCoco, 'r') as f:
    classes = f.read().splitlines()

#Adding 'salt and pepper' to an image using probability

def sp_noise(img,prob):
    output = np.zeros(img.shape,np.uint8)
    upper = 1 - prob 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > upper:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    return output

#Reading an image from file directory

img = cv2.imread('D:/Users/LeeHoang/Downloads/Wildtrack_dataset_full/Wildtrack_dataset/Image_subsets/C6/00000000.png')


currentFrame = []
#read the current frame from the video
_ , img = cap.read()
height, width, _ = img.shape
#Load camera matrix and remove distortion
rm, tm, cm, dc = returnCameraParameters(6)
img = cv2.undistort(img,cm,dc ,None, cm)
#Forward the image
blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0), swapRB=True, crop =False)
#net.setInput is used to add an input to the network.
net.setInput(blob)
#Getting the output layer's names
output_layers_names = net.getUnconnectedOutLayersNames()
tic = timeit.default_timer()
#passing the input through the layers.
layerOutputs = net.forward(output_layers_names)
#Boxes record the coordinates of the bounding boxes
boxes = []

#Confidence records the confidence percentage of the object detected
confidences = []
#ClassID records the 
class_ids = []
for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            boxes.append([x,y,w,h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)
#This function applies the bounding boxes found in the previous for loop without the use of Non max suppression
NONMS = False
if (NONMS):
    count = 0
    font = cv2.FONT_HERSHEY_PLAIN
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(img, (startX, startY), (startX+endX, startY+endY), (0, 0, 255), 2)
        label = str(classes[class_ids[count]])
        confidence = str(round(confidences[count],2)) 
        cv2.putText(img, label + " " + confidence, (startX,startY + 20), font , 2, (255,255,255), 2)
        count += 1

#Non max suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.5)
#filterPeople adds the bounding box coordinates to a list
filterPeople = []
colours = np.random.uniform(0,255, size = (len(boxes), 3))
if len(indexes)>0:
    for i in indexes.flatten():
        if (str(classes[class_ids[i]]) == "person"):
            filterPeople.append(boxes[i])
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            colour = colours[i]
#If the system runs through the first frame of the video, the positions will be added
if not previousFrame:
    for x in range(len(filterPeople)):
        previousFrame.append([int(filterPeople[x][0]+(filterPeople[x][2]*0.5)),int(filterPeople[x][1]+filterPeople[x][3]*0.99)])
#Add current positions
for x in range(len(filterPeople)):
    currentFrame.append([int(filterPeople[x][0]+(filterPeople[x][2]*0.5)),int(filterPeople[x][1]+filterPeople[x][3]*0.99)])
#Selection of calculating social distancing
if (method == "Homography"):
    #Blackbox for top down view.
    blackBox = cv2.imread('D:\year3\IN3007-IndividualProject/blackbox.png')
    #resize black box to concatenate with img later
    blackBox = cv2.resize(blackBox, (1920,1080))
    img,blackBox,stablizer,sBool, notSD = SDCalibratedCameraWithNoZ(img,blackBox,filterPeople,stablizer,sBool)
    blackBox = cv2.resize(blackBox, (800,1080))
    img = np.concatenate((img,blackBox),axis=1)
elif (method == "EstimateHomography"):
    blackBox = cv2.imread('D:\year3\IN3007-IndividualProject/blackbox.png')
    blackBox = blackBox
    img = SDCalibratedCamera(img,filterPeople)
    img = cv2.resize(img, (1200,668))
    img = np.concatenate((img,blackBox),axis=1)
elif (method == "MatrixTransformation"):
    blackBox = cv2.imread('D:\year3\IN3007-IndividualProject/blackbox.png')
    blackBox = cv2.resize(blackBox, (2600,1270))
    #Points in the image to transform
    pt1 = np.float32([[55,209],[590,170],[2600,526],[423,1270]])
    pt2 = np.float32([[0,0],[2600,0],[2600,1270],[0,1270]])
    #Create matrix transformation and transform image
    matrix = cv2.getPerspectiveTransform(pt1,pt2)
    result = cv2.warpPerspective(img,matrix, (2600,1270))
    result, greenPeople, redPeople = SDMatrixTransformation(blackBox,filterPeople,matrix)
    #Mark people who are social distancing as green, otherwise red
    for i in greenPeople:
        cv2.circle(img,(filterPeople[i][0]+int(filterPeople[i][2]/2),filterPeople[i][1]+int(filterPeople[i][3]*0.99)),20,(0,255,0), 3)
    
    for i in redPeople:
        cv2.circle(img,(filterPeople[i][0]+int(filterPeople[i][2]/2),filterPeople[i][1]+int(filterPeople[i][3]*0.99)),20,(0,0,255), 3)
    
    blackBox = cv2.resize(blackBox, (500,668))
    img = cv2.resize(img, (1200,668))
    result = cv2.resize(result, (500,668))
    img = np.concatenate((img,result),axis=1)
elif(method == "SDBoundingBox"):
    img = SDBoundingBox(img,filterPeople)
    img = cv2.resize(img, (1200,668))
#This is to enable the 'group' detection        
if(GD):
    img, peopleMap, peopleWhoAreGrouped,uniquePeople = GroupDetection(img,filterPeople, previousFrame,currentFrame, peopleMap, notSD, peopleWhoAreGrouped, uniquePeople)
    img = cv2.resize(img, (1200,668))
#PreviousFrame is now the current frame as we are going to the next frame.
previousFrame = currentFrame
#Resize image to fit on a screen
img = cv2.resize(img, (1700,668))
#Show the current frame
cv2.imshow('result', img)
#Record the current frame
out.write(img)


cv2.imshow('Image', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
