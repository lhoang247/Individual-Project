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


cv2.rectangle(img, (293,116), (325, 239), (255,0,255), 2)
cv2.rectangle(img, (400,119), (433, 250), (255,0,255), 2)
cv2.rectangle(img, (344,127), (386, 289), (255,0,255), 2)
cv2.rectangle(img, (1222,152), (1287, 381), (255,0,255), 2)
cv2.rectangle(img, (31,116), (68, 250), (255,0,255), 2)
'''
#Data augmentation

#Adding noise to the chosen img

img = sp_noise(img,0.05)

#Resizing image

img = cv2.resize(img,(1280,720))
'''
blackBox = cv2.imread('D:\year3\IN3007-IndividualProject/blackbox.png')

#Getting the resolution of the image

height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0), swapRB=True, crop =False)

#net.setInput is used to add an input to the network.

net.setInput(blob)

#Getting the output layer's names
output_layers_names = net.getUnconnectedOutLayersNames()

tic = timeit.default_timer()

#passing the input through the layers.
layerOutputs = net.forward(output_layers_names)
confidence_test = []
loopcount = []
boxes = []
confidences = []
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
confidence_test.append(len(boxes))
'''
plt.title("Objects detected after confidence filtering")
plt.locator_params(nbins = 20)
plt.plot(loopcount,confidence_test)
plt.xlabel("Confidence (%)")
plt.ylabel("Objects left after filtering")
plt.show()
print(confidence_test)
'''
toc = timeit.default_timer()

print("here: " , toc - tic)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.9, 0.5)

font = cv2.FONT_HERSHEY_PLAIN

colours = np.random.uniform(0,255, size = (len(boxes), 3))


'''cv2.rectangle(img,(1510,139),(1561,299),(255,255,255),1)
cv2.rectangle(img,(1273,129),(1319,282),(0,255,255),1)
cv2.rectangle(img,(69,147),(152,378),(0,255,255),1)
'''

filterPeople = []

if len(indexes)>0:
    for i in indexes.flatten():
        if (str(classes[class_ids[i]]) == "person"):
            filterPeople.append(boxes[i])
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
    
            colour = colours[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)



img = SDBoundingBox(img,filterPeople)

count = 0

'''cv2.circle(img,(0,1000),5,(255,0,0), -1)
cv2.circle(img,(1700,700),5,(255,0,0), -1)
cv2.circle(img,(300,500),5,(255,0,0), -1)
cv2.circle(img,(980,450),5,(255,0,0), -1)
'''
pt1 = np.float32([[0,0],[1700,700],[1000,500],[980,450]])
pt2 = np.float32([[0,0],[1040,0],[0,668],[1040,668]])

matrix = cv2.getPerspectiveTransform(pt1,pt2)

print("MAtrix = ", matrix)

result = cv2.warpPerspective(img,matrix, (1040,668))

'''
for (startX, startY, endX, endY) in boxes:
    cv2.rectangle(img, (startX, startY), (startX+endX, startY+endY), (0, 0, 255), 2)
    label = str(classes[class_ids[count]])
    confidence = str(round(confidences[count],2))
    cv2.putText(img, label + " " + confidence, (startX,startY + 20), font , 2, (255,255,255), 2)
    count += 1
'''


'''for b in blob:
    for n, img_blob in enumerate(b):
        cv2.imshow(str(n), img_blob)
'''

SDMatrixTransformation(img,filterPeople,matrix)


cameraMatrix = np.matrix([1743.4478759765625,0.0,934.5202026367188,0.0,1735.1566162109375,444.3987731933594,0.0,0.0,1.0])
cameraMatrix = np.reshape(cameraMatrix, (3,3))
print(np.linalg.inv(cameraMatrix))

distortionCoefficient = np.array([0.43248599767684937,0.6106230020523071,0.008233999833464622,0.0018599999602884054,-0.6923710107803345])
print(distortionCoefficient)

rvec =  np.matrix([1.759099006652832,0.46710100769996643 ,-0.331699013710022])
tvec = np.matrix([525.8941650390625 ,45.40763473510742 ,986.7235107421875])
R_mat = np.transpose(cv2.Rodrigues(rvec)[0].reshape(3,3))
print(R_mat)


dst = cv2.warpPerspective(img,cameraMatrix,(1000,1000))

realWorld = np.array([[0.9750000000003638,17.00828125],[326.15000000000146,14.685729166666668],[444.875,9.933072916666667],[330.375,0.6945312500000007]])
imagePoints = np.array([[309, 115],[365, 118],[416, 127],[1254, 151]])

h, status = cv2.findHomography(imagePoints, realWorld)

img = cv2.resize(img, (1024,668))
blackBox = cv2.resize(blackBox, (500,668))
blackBox = blackBox - 1
blackBox = SDCalibratedCamera(blackBox,filterPeople)
combined = np.concatenate((img,blackBox),axis=1)


cv2.imshow('Image', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
