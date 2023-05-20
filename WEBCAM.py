import cv2
from skimage.feature import hog
from skimage.feature import local_binary_pattern
import numpy as np
from sklearn.svm import SVC
import joblib


class image:
    def __init__(self, img):
        self.img = img

    def cropImg(self):
        grayImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        faceClassifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        face = faceClassifier.detectMultiScale(
            grayImg, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        imgRgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        for x, y, w, h in face:
            imgRgb = imgRgb[y : y + h, x : x + w]
            break
        self.img = imgRgb

    def hogFeature(self):
        self.img = cv2.resize(self.img, (180, 180))
        fd, hogImg = hog(
            self.img,
            orientations=30,
            pixels_per_cell=(30, 30),
            cells_per_block=(1, 1),
            visualize=True,
            channel_axis=-1,
        )
        return fd

    def lbpFeature(self):
        grayImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(grayImg, 24, 3, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(24), range=(0, 26))
        hist = hist.astype("float")
        hist /= hist.sum() + 1e-7
        return hist


if __name__ == "__main__":
    vid = cv2.VideoCapture(0)

    filename = "myModel.joblib"
    loadedModel = joblib.load(filename)


