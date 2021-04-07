import numpy as np
import cv2
import os
import timeit
import matplotlib.pyplot as plt
from SDBoundingBox import SDBoundingBox
from MatrixTransformation import SDMatrixTransformation
from SDEstimatedCalibratedCamera import SDCalibratedCamera
from HomographyMethod import SDCalibratedCameraWithNoZ
from CameraMatrix import returnCameraParameters
from GroupDetection import GroupDetection
def VideoPlayer(method):
    dir = 'D:\year3\IN3007-IndividualProject\YOLOv3'

    pathYoloWeights = dir + '\yolov3.weights'
    pathYoloCfg =  dir + '\yolov3.cfg'
    pathCoco =  dir + '\coco.names'

    net = cv2.dnn.readNet(pathYoloWeights, pathYoloCfg)

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    classes = []

    sBool = False
    with open(pathCoco, 'r') as f:
        classes = f.read().splitlines()

    stablizer = []
    previousFrame = []
    peopleMap = []
    cap = cv2.VideoCapture('D:/year3/IN3007-IndividualProject/videos used/cam3clip1.mp4')
    tic = timeit.default_timer()
    while True:

        currentFrame = []

        _ , img = cap.read()
        height, width, _ = img.shape
        rm, tm, cm, dc = returnCameraParameters(2)
        img = cv2.undistort(img,cm,dc ,None, cm)
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
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.5)

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

        if not previousFrame:
            for x in range(len(filterPeople)):
                previousFrame.append([int(filterPeople[x][0]+(filterPeople[x][2]*0.5)),int(filterPeople[x][1]+filterPeople[x][3]*0.99)])

        for x in range(len(filterPeople)):
            currentFrame.append([int(filterPeople[x][0]+(filterPeople[x][2]*0.5)),int(filterPeople[x][1]+filterPeople[x][3]*0.99)])

        if (method == "Homography"):
            blackBox = cv2.imread('D:\year3\IN3007-IndividualProject/blackbox.png')
            blackBox = cv2.resize(blackBox, (1920,1080))
            img,blackBox,stablizer,sBool = SDCalibratedCameraWithNoZ(img,blackBox,filterPeople,stablizer,sBool)
            img = cv2.resize(img, (1200,668))
            blackBox = cv2.resize(blackBox, (500,668))
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
            pt1 = np.float32([[55,209],[590,170],[2600,526],[423,1270]])
            pt2 = np.float32([[0,0],[2600,0],[2600,1270],[0,1270]])
            matrix = cv2.getPerspectiveTransform(pt1,pt2)
            result = cv2.warpPerspective(img,matrix, (2600,1270))
            result, greenPeople, redPeople = SDMatrixTransformation(blackBox,filterPeople,matrix)
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
        elif(method == "GroupDetection"):
            img, peopleMap = GroupDetection(img,filterPeople, previousFrame,currentFrame, peopleMap)
            img = cv2.resize(img, (1200,668))

        previousFrame = currentFrame

        cv2.imshow('result', img)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    toc = timeit.default_timer()
    print(toc - tic)