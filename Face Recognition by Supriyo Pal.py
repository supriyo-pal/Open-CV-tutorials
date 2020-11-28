# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 22:20:30 2020

@author: Supriyo
"""

import cv2
cap=cv2.VideoCapture(0)
cascade_classifier = cv2.CascadeClassifier('haarcascades_frontalface_default.xml')
while True:
    ret,frame=cap.read()
    frame=cv2.cvtColor(frame,0)
    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    detections=cascade_classifier.detectMultiScale(gray_img,1.3,5)
    
    if (len(detections)>1):
        (x,y,w,h)=detections[0]
        frame =cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('Face Recognition By Supriyo Pal',frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
    
    #cv2.waitKey(1)# 1 is milisec
cap.release()
cv2.destroyAllWindows()
    