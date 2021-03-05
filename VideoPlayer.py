import numpy as np
import cv2
import os
import timeit
import matplotlib.pyplot as plt
dir = 'D:\year3\IN3007-IndividualProject\YOLOv3'

pathYoloWeights = dir + '\yolov3.weights'
pathYoloCfg =  dir + '\yolov3.cfg'
pathCoco =  dir + '\coco.names'

net = cv2.dnn.readNet(pathYoloWeights, pathYoloCfg)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = []

with open(pathCoco, 'r') as f:
    classes = f.read().splitlines()


cap = cv2.VideoCapture('D:/year3/IN3007-IndividualProject/videos used/cam2clip1.mp4')
tic = timeit.default_timer()
while True:
    _ , img = cap.read()
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

    '''
    count = 0
    font = cv2.FONT_HERSHEY_PLAIN
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(img, (startX, startY), (startX+endX, startY+endY), (0, 0, 255), 2)
        label = str(classes[class_ids[count]])
        confidence = str(round(confidences[count],2))
        cv2.putText(img, label + " " + confidence, (startX,startY + 20), font , 2, (255,255,255), 2)
        count += 1
    '''

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.6)

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
                cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

    img = cv2.resize(img, (1024,668))

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
toc = timeit.default_timer()
print(toc - tic)