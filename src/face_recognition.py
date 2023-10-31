import cv2
import numpy as np
import pandas as pd
import face_recognition
import os
import pickle
from datetime import datetime
# from PIL import ImageGrab
 
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 
def markAttendance(name, df):
    if name not in df['Name'].values:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        #df = df.append({'Name':name, 'Count':0, 'Time':dtString, 'Status':'A'}, ignore_index = True)
        df.loc[len(df.index)] = [name, 0, dtString, 'A'] 
    else:
        first_time=datetime.now()
        duration=round(df.loc[df['Name'] == name, 'Duration'], 2)
        later_time = datetime.now()
        difference = later_time - first_time
        seconds_in_day = 24 *60*60
        time_spent=int(divmod(difference.days * seconds_in_day + difference.seconds, 60)[1])
        dtString = later_time.strftime('%H:%M:%S')
        df.loc[df['Name'] == name, 'Duration']=round(duration+0.34, 2)
        if int(duration+time_spent)>=10:
            df.loc[df['Name'] == name, 'Status']='P' 
                
                
        
 
#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr
 
#encodeListKnown = findEncodings(images)
#print('Encoding Complete')
 
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnown = pickle.load(file)
file.close()
# print(studentIds)
print("Encode File Loaded")

cap = cv2.VideoCapture(0)
df = pd.read_csv("Attendance.csv")
while True:
    success, img = cap.read()
    #img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name, df)
 
    cv2.imshow('Webcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
df.to_csv("Attendance.csv", index=False)
cv2.destroyAllWindows()
