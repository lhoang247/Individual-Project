import numpy as np
import cv2
import os

dir = 'D:\year3\IN3007-IndividualProject\YOLOv3'

pathYoloWeights = dir + '\yolov3.weights'
pathYoloCfg =  dir + '\yolov3.cfg'
pathCoco =  dir + '\coco.names'

net = cv2.dnn.readNet(pathYoloWeights, pathYoloCfg)

classes = []

with open(pathCoco, 'r') as f:
    classes = f.read().splitlines()

img = cv2.imread('D:\year3\IN3007-IndividualProject\Video Projects/00000730.png')

height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0), swapRB=True, crop =False)
#net.setInput is used to add an input to the network.
net.setInput(blob)

#Getting the output layer's names
output_layers_names = net.getUnconnectedOutLayersNames()

#passing the input through the layers.
layerOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
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



print(boxes)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

print(indexes.flatten())

font = cv2.FONT_HERSHEY_PLAIN

colours = np.random.uniform(0,255, size = (len(boxes), 3))

for i in indexes.flatten():
    if (str(classes[class_ids[i]]) == "person"):
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i],2))
        colour = colours[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), colour, 2)
        cv2.putText(img, label + " " + confidence, (x,y + 20), font , 2, (255,255,255), 2)


'''for b in blob:
    for n, img_blob in enumerate(b):
        cv2.imshow(str(n), img_blob)
'''

img = cv2.resize(img, (1024,668))

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(classes)