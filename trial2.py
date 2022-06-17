import numpy as np
import cv2

f_cascade = cv2.CascadeClassifier("Scripts/haarcascade_frontalface_alt_tree.xml")
# e_cascade = cv2.CascadeClassifier("eye.xml")

image = cv2.imread("Project_extra/Sid-Roy.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = f_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('img', image)
cv2.waitKey(0)
cv2.destroyAllWindows()