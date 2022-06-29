# Required Libraries imported 
import cv2
import numpy as np

class LaneDetector:
    #------------------------- Constructor -------------------------#
    def __init__(self, Img, Path):
        self._IMG = Img
        self._PATH = Path
        self._LENGTH = len(self._PATH)

    def __make_points(image, average):
        slope, y_int = average
        y1 = image.shape[0]
        y2 = int(y1 * (3/5))
        Y1, Y2 = y1 - y_int, y2 - y_int
        x1 = int(Y1 // slope)
        x2 = int(Y2 // slope)
        return np.array([x1, y1, x2, y2])

    #Display lane lines
    def __display_lines(image, lines):
        lines_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return lines_image
    
    def detectionOfLane(self):
        if self._PATH[self._LENGTH - 3:] == "png" or self._PATH[self._LENGTH - 3:] == "bmp" or self._PATH[self._LENGTH - 3:] == "jpg" or self._PATH[self._LENGTH - 4:] == "jpeg":
            grey = cv2.cvtColor(self._IMG, cv2.COLOR_BGR2GRAY)
            gauss = cv2.GaussianBlur(grey, (5, 5), 0)
            
            lap = cv2.Laplacian(grey, cv2.CV_64F)
            lap = np.uint8(np.absolute(lap))
            # self.imgDDisp(self._WIN_NAME, lap)
            # Sobel and Gradient Detction logic
            sobel = cv2.Sobel(grey, cv2.CV_64F, 1, 0)
            sobel_Y = cv2.Sobel(grey, cv2.CV_64F, 0, 1)
            sobel_X = np.uint8(np.absolute(sobel_X))
            sobel_Y = np.uint8(np.absolute(sobel_Y))
            edges = cv2.bitwise_or(sobel_X, sobel_Y)

            lines = cv2.HoughLinesP(edges, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 3)
            
            left = []
            right = []
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope = parameters[0]
                y_int = parameters[1]
                if slope < 0:
                    left.append((slope, y_int))
                else:
                    right.append((slope, y_int))
            #Average of slopes and y-intercept
            right_avg = np.average(right, axis = 0)
            left_avg = np.average(left, axis = 0)
            left_line = self.__make_points(self._IMG, left_avg)
            right_line = self.__make_points(self._IMG, right_avg)
            averaged_lines = [left_line, right_line]

            black_lines = self.__display_lines(self._IMG, averaged_lines)
            lanes = cv2.addWeighted(self._, 0.8, black_lines, 1, 1)
            cv2.imshow("lanes", lanes)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return lanes