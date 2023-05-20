import random
import cv2
from skimage.feature import hog
from skimage.feature import local_binary_pattern
import numpy as np

class image:
    def __init__(self, path):
        self.img = cv2.imread(path)

    def cropImg(self):
        grayImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        face = faceClassifier.detectMultiScale(grayImg, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        imgRgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        for (x, y, w, h) in face:
            imgRgb = imgRgb[y:y+h, x:x+w]
            break
        self.img = imgRgb
    
    def hogFeature(self):
        resizedImg = cv2.resize(self.img, (180, 180))
        fd, hog_image = hog(resizedImg, orientations=30, pixels_per_cell=(30, 30), cells_per_block=(1, 1), visualize=True, channel_axis=-1)
        return fd


class svm:
    def divide(self):
        self.test = set()

        while len(self.test) < (4000 / 100 * 30):
            self.test.add(random.randint(1, 4000))
        
        # train = z(1, 4000) - test
    
    def readLabels(self):
        self.labels = []

        with open("labels.txt") as FILE:
            for line in FILE:  
                self.labels.append(int(line[0]))