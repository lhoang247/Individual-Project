import numpy as np
import cv2
import os
import timeit
dir = 'D:\year3\IN3007-IndividualProject\YOLOv3'

pathYoloWeights = dir + '\yolov3.weights'
pathYoloCfg =  dir + '\yolov3.cfg'
pathCoco =  dir + '\coco.names'

net = cv2.dnn.readNet(pathYoloWeights, pathYoloCfg)

classes = []

with open(pathCoco, 'r') as f:
    classes = f.read().splitlines()

img = cv2.imread('D:\year3\IN3007-IndividualProject\images used/00000000.png')
print(img.shape)
blackBox = cv2.imread('D:\year3\IN3007-IndividualProject/blackbox.png')
height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0), swapRB=True, crop =False)
#net.setInput is used to add an input to the network.
net.setInput(blob)

#Getting the output layer's names
output_layers_names = net.getUnconnectedOutLayersNames()

tic = timeit.default_timer()

#passing the input through the layers.
layerOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

print()

for output in layerOutputs:
    print(len(output))
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            
            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x,y,w,h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

toc = timeit.default_timer()

print(toc - tic)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.67, 0.5)

font = cv2.FONT_HERSHEY_PLAIN

colours = np.random.uniform(0,255, size = (len(boxes), 3))

cv2.rectangle(img,(1510,139),(1561,299),(255,255,255),1)
cv2.rectangle(img,(1273,129),(1319,282),(0,255,255),1)
cv2.rectangle(img,(69,147),(152,378),(0,255,255),1)

for i in indexes.flatten():
    if (str(classes[class_ids[i]]) == "person"):
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i],2))
  
        colour = colours[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), colour, 2)
        cv2.putText(img, label + " " + confidence, (x,y + 20), font , 2, (255,255,255), 2)

count = 0

cv2.circle(img,(0,1000),5,(255,0,0), -1)
cv2.circle(img,(1700,700),5,(255,0,0), -1)
cv2.circle(img,(300,500),5,(255,0,0), -1)
cv2.circle(img,(980,450),5,(255,0,0), -1)

pt1 = np.float32([[0,1000],[1700,700],[300,500],[980,450]])
pt2 = np.float32([[0,0],[700,0],[0,600],[700,600]])

matrix = cv2.getPerspectiveTransform(pt1,pt2)

result = cv2.warpPerspective(img,matrix, (700,600))

'''for (startX, startY, endX, endY) in boxes:
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

cameraMatrix = np.matrix([1743.4478759765625,0.0,934.5202026367188,0.0,1735.1566162109375,444.3987731933594,0.0,0.0,1.0])
cameraMatrix = np.reshape(cameraMatrix, (3,3))
print(np.linalg.inv(cameraMatrix))

distortionCoefficient = np.array([0.43248599767684937,0.6106230020523071,0.008233999833464622,0.0018599999602884054,-0.6923710107803345])
print(distortionCoefficient)

rvec =  np.matrix([1.759099006652832,0.46710100769996643 ,-0.331699013710022])
tvec = np.matrix([525.8941650390625 ,45.40763473510742 ,986.7235107421875])

print("here")

print(tvec - 1381*np.linalg.pinv(rvec*cameraMatrix)*np.matrix([1510,139,1]))

dst = cv2.warpPerspective(img,cameraMatrix,(1000,1000))


img = cv2.resize(img, (1024,668))
blackBox = cv2.resize(blackBox, (500,668))
blackBox = blackBox - 1
combined = np.concatenate((img,blackBox),axis=1)
cv2.rectangle(combined,(1300,300),(1500,500), (0, 0, 255), 2)



cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
