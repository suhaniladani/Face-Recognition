from facerec.feature import Fisherfaces
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.serialization import save_model, load_model
import numpy as np
from PIL import Image
from PIL import ImageChops
import sys, os
import time
import cv2
import multiprocessing


#vc=cv2.VideoCapture('Videos/Video8.mp4')
vc=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

pathdir='FaceDB/'

if not os.path.exists(pathdir): os.makedirs(pathdir)
print ( 'Are you ready for me to take some pictures ? \n ')
print ( ' It will only take 5 seconds \ n press " S " when you are in the middle ')
while (1):
	ret,frame = vc.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 3)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('Recognition',frame)

        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
cv2.destroyAllWindows()
    
start = time.time()
count = 0
while int(time.time()-start) <=10:
        ret,frame = vc.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 3)
        for (x,y,w,h) in faces:
            cv2.putText(frame,'Click!', (x,y), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,250),3,1)
            count +=1
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            resized_image = cv2.resize(frame[y:y+h,x:x+w], (273, 273))
            if count%10 == 0:
                print  pathdir+str(time.time()-start)+'.jpg'
                cv2.imwrite( pathdir+'/'+str(time.time()-start)+'.jpg', resized_image );
        cv2.imshow('Recognition',frame)
        cv2.waitKey(10)
cv2.destroyAllWindows()

