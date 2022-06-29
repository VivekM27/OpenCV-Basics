# Importing required libraries.
import cv2
import numpy as np

class machineDetection:
    #------------------------- Constructor -------------------------#
    def __init__(self, Img, Path):
        self._IMG = Img
        self._PATH = Path
        self._LENGTH = len(self._PATH)
        # GrayScale
        self._GREY = cv2.cvtColor(self._IMG, cv2.COLOR_BGR2GRAY)

    def carDetector(self, ch):
        if self._PATH[self._LENGTH - 3:] == "png" or self._PATH[self._LENGTH - 3:] == "bmp" or self._PATH[self._LENGTH - 3:] == "jpg" or self._PATH[self._LENGTH - 4:] == "jpeg":
            # Blurr
            blur = cv2.GaussianBlur(self._GREY,(5,5),0)

            # Dilation
            dilated = cv2.dilate(blur,np.ones((3,3)))

            # Morphology transformation with kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

            # Car Cascade
            car_cascade_src = 'Scripts/cars.xml'
            car_cascade = cv2.CascadeClassifier(car_cascade_src)
            cars = car_cascade.detectMultiScale(closing, 1.1, 1)

            # Counting number of cars
            cnt = 0
            for (x, y, w, h) in cars:
                cv2.rectangle(self._IMG, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cnt += 1

            if ch == "image":
                return self._IMG
            else:
                return cnt

    def busDetector(self, ch):
        if self._PATH[self._LENGTH - 3:] == "png" or self._PATH[self._LENGTH - 3:] == "bmp" or self._PATH[self._LENGTH - 3:] == "jpg" or self._PATH[self._LENGTH - 4:] == "jpeg":
            # Blurr
            blur = cv2.GaussianBlur(self._GREY,(5,5),0)

            # Dilation
            dilated = cv2.dilate(blur,np.ones((3,3)))

            # Morphology transformation with kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

            # Bus Cascade
            bus_cascade_src = 'Scripts/Bus_front.xml'
            bus_cascade = cv2.CascadeClassifier(bus_cascade_src)
            bus = bus_cascade.detectMultiScale(closing, 1.1, 1)

            # Counting number of buses
            cnt = 0
            for (x,y,w,h) in bus:
                cv2.rectangle(self._IMG,(x,y),(x+w,y+h),(255,0,0),2)
                cnt += 1

            if ch == "image":
                return self._IMG
            else:
                return cnt

# ########################
# # Detection for videos #
# ########################
# cascade_src = 'cars.xml'
# video_src = 'Cars.mp4'

# cap = cv2.VideoCapture(video_src)
# car_cascade = cv2.CascadeClassifier(cascade_src)
# video = cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (450,250))  

# # Now we will read frames one by one from the input video, 
# #   convert them into grayscale and using car cascade to detect all cars in that particular frame. 
# #   In the end we write this video using video.write() method and video.release() will save this video to the given path. 
# while True:
#     ret, img = cap.read()
   
#     if (type(img) == type(None)):
#         break
        
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cars = car_cascade.detectMultiScale(gray, 1.1, 2)

#     for (x,y,w,h) in cars:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)

# video.write(img) 
# video.release()