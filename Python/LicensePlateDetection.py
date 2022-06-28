# Importing Libraries
import cv2
import imutils
import pytesseract
import numpy as np

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'D:\Developement\\Tesseract\\tesseract.exe'

class LPD:
    def __init__(self, Img, Path):
        self._IMG = Img
        self._PATH = Path
        self._LENGTH = len(self._PATH)
        self._PLATE = ""

    def licensePD(self):
        if self._PATH[self._LENGTH - 3:] == "png" or self._PATH[self._LENGTH - 3:] == "bmp" or self._PATH[self._LENGTH - 3:] == "jpg" or self._PATH[self._LENGTH - 4:] == "jpeg":
            # Conversion to Greyscale
            self.__GRAY_IMG = cv2.cvtColor(self._IMG, cv2.COLOR_BGR2GRAY)

            # Image Noise Reduction
            self.__GRAY_IMG = cv2.bilateralFilter(self.__GRAY_IMG, 11, 17, 17)

            # Edge Detection and Smoothing
            self.__EDGED = cv2.Canny(self.__GRAY_IMG, 30, 200)

            # Contours Detection from Edged Detected Image
            cnts, new = cv2.findContours(self.__EDGED.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            image1 = self._IMG.copy()
            # cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)

            # Sorting identified Contours - minimum point to ignore - 30 - ignoring below 30
            cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:30]
            screenCnt = None
            image2 = self._IMG.copy()
            # cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)

            # Detecting Contours in Boxed Format
            i = 7
            for c in cnts:
                perimeter = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
                if len(approx) == 4: 
                    screenCnt = approx
                x,y,w,h = cv2.boundingRect(c)

                # Crop out detected image
                new_img = self._IMG[y : y + h, x : x + w]
                cv2.imwrite('Images/license.png', new_img)
                i += 1
                break

            # Pointing out contour drawn on license plate
            # cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)

            # Detected License plate
            Cropped_loc = cv2.imread('Images/license.png')
            self._PLATE = pytesseract.image_to_string(Cropped_loc, lang = 'eng')
            # cv2.imshow(plate, cv2.imread(Cropped_loc))
            return Cropped_loc
        else:
            print("Cannot run for unsupported file types!!!\n please provide file with \".bmp, .jpg, .jpeg or .png\" extension")
            print("grayImg() stopped!!!", end = "\n----------------------\n")
            return np.zeros((300, 300, 3), dtype = "uint8")

    def plateNumber(self):
        if self._PATH[self._LENGTH - 3:] == "png" or self._PATH[self._LENGTH - 3:] == "bmp" or self._PATH[self._LENGTH - 3:] == "jpg" or self._PATH[self._LENGTH - 4:] == "jpeg":
            return self._PLATE
        else:
            print("Cannot run for unsupported file types!!!\n please provide file with \".bmp, .jpg, .jpeg or .png\" extension")
            print("plateNumber() stopped!!!", end = "\n----------------------\n")
            return ""