
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(cv2.CAP_PROP_FPS,60)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()
imgBg = cv2.imread('img2.jpeg')
while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img,imgBg,threshold= 0.8)
    imageStacked = cvzone.stackImages([img, imgOut],2,1)
    _, imageStacked = fpsReader.update(imgOut, color=(0,255,0))
    cv2.imshow("Image Out",imageStacked)
    cv2.waitKey(1)