import random
import cv2

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