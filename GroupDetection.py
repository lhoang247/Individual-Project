import cv2
import numpy as np
import matplotlib.pyplot as plt

def GroupDetection(img, filterPeople, previousFrame, currentFrame, peopleMap, notSD, peopleWhoAreGrouped, uniquePeople):
    
    #for x in range(len(filterPeople)):
    #    cv2.rectangle(img, (filterPeople[x][0],filterPeople[x][1]), (filterPeople[x][0]+filterPeople[x][2], filterPeople[x][1]+filterPeople[x][3]), (255,0,0), 2)

    #If map is empty then add people to the map
    if not peopleMap:
        for i in range(len(currentFrame)):
            peopleMap.append([i,currentFrame[i][0],currentFrame[i][1],i])
            uniquePeople += 1

    #Newcomers is a list of people that do not have a key already.
    newcomers = []

    #peopleMapReoccurance is a list of people who have a key already.
    peopleMapReoccurance = []

    #This for loop tries to detect people with keys and updates their position
    for i in range(len(currentFrame)):
        indentified = False
        for j in range(len(peopleMap)):
            if (abs(currentFrame[i][0] - peopleMap[j][1]) < 50 and abs(currentFrame[i][1] - peopleMap[j][2]) < 50):
                peopleMap[j][1] = currentFrame[i][0]
                peopleMap[j][2] = currentFrame[i][1]
                peopleMap[j][3] = i
                indentified = True
                peopleMapReoccurance.append(j)
        if not indentified:
            newcomers.append([currentFrame[i][0],currentFrame[i][1]])
    

    #PeopleMapTemp 
    peopleMapTemp = []

    #Record the poeple who still maintain their key.
    for occurance in peopleMapReoccurance :
        peopleMapTemp.append(peopleMap[occurance])

    peopleMap = peopleMapTemp

    #Add people who are new comers to the peopleMap
    for i in range(len(newcomers)):
        peopleMap.append([uniquePeople + 1,newcomers[i][0],newcomers[i][1],-1])
        uniquePeople += 1


    font = cv2.FONT_HERSHEY_PLAIN


    #Group list is a list that groups people who are not social distancing together.
    groupList = []
    
    #This for loop sorts the duplicate of items and removes them
    for i in range(len(notSD)):
        if notSD[i][0] > notSD[i][1]:
            notSD[i] = [notSD[i][1],notSD[i][0]]

    sortedNotSD = []

    #Remove duplicate of items.
    for i in notSD:
        if i not in sortedNotSD:
            sortedNotSD.append(i)

    #Sort by the first element of each item in the list.
    def takeFirst(elem):
        return elem[0]

    sortedNotSD.sort(key = takeFirst)

    frameToMap = []

    #Map output of object detector with peopleMap
    for i in sortedNotSD:
        foundPartner = False
        for j in range(len(frameToMap)):
            for k in frameToMap[j]:
                if (i[0] == k):
                    temp = j
                    foundPartner = True
        
        if foundPartner:
            if i[1] not in frameToMap[j]:
                frameToMap[j].append(i[1])
        else:
            frameToMap.append([i[0],i[1]])

    #Link the people from the object detector output.
    for i in frameToMap:
        for j in range(len(i)):
            for k in peopleMap:
                if (i[j] == k[3]):
                    i[j] = k[0]


    #Sort each element in frameToMap
    for i in frameToMap:
        i.sort()

    #If empty
    if not peopleWhoAreGrouped:
        peopleWhoAreGrouped = frameToMap

    print(frameToMap)

    #for i in range(len(notSD)):
    #    if(notSD[i][0])

    for i in range(len(peopleMap)):
        cv2.putText(img,str(peopleMap[i][0]), (peopleMap[i][1],peopleMap[i][2] - 50), font , 2, (0,127,255), 2)
    
    [[1][1]]

    for i in frameToMap:
        for k in range(len(i)):
            for l in peopleMap:
                if l[0] == i[k] and (l[1] < 400 or l[1] > 1520) and (l[2] < 400 or l[2] > 680):
                    if i not in peopleWhoAreGrouped:
                        peopleWhoAreGrouped.append(i)


    for x in peopleWhoAreGrouped:
        for y in range(len(x)):
            for z in peopleMap:
                if x[y] == z[0]:
                    cv2.rectangle(img, (filterPeople[z[3]][0],filterPeople[z[3]][1]), (filterPeople[z[3]][0]+filterPeople[z[3]][2], filterPeople[z[3]][1]+filterPeople[z[3]][3]), (0,255,0), 10)
    
    return img, peopleMap, peopleWhoAreGrouped, uniquePeople