# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time

def frame_sub(img1, img2, img3, th):
    diff1 = cv2.absdiff(img2, img1)
    diff2 = cv2.absdiff(img3, img2)
    
    diff = cv2.bitwise_and(diff1, diff2)
    diff[diff < th] = 0
    diff[diff >= th] = 255
    
    mask = cv2.medianBlur(diff, 3)
    
    return mask

cap = cv2.VideoCapture("C:/Users/matsuzaki/Desktop/卒業研究/水槽試験MP4/ikesu1.mp4")

frame1 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
frame2 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
frame3 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

while(cap.isOpened()):
    mask = frame_sub(frame1, frame2, frame3, th=3)
    
    cv2.imshow("Capture_result", frame2)
    cv2.imshow("Mask", mask)
    
    frame1 = frame2
    frame2 = frame3
    frame3 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    
    time.sleep(0.05)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()