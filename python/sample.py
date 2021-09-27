import numpy as np
# Import OpenCV for easy image rendering
import cv2
import math
from matplotlib import pyplot as plt
import csv
import sys

def red_detect(image):
    hsvLower = np.array([160,64,0])    # 抽出する色の下限(HSV)
    hsvUpper = np.array([179,255,255])    # 抽出する色の上限(HSV)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # 画像をHSVに変換
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)    # HSVからマスクを作成
    hsvLower2 = np.array([0,64,0])    # 抽出する色の下限(HSV)
    hsvUpper2 = np.array([30,255,255])    # 抽出する色の上限(HSV)
    hsv_mask2 = cv2.inRange(hsv, hsvLower2, hsvUpper2)    # HSVからマスクを作成
    mask=hsv_mask+hsv_mask2
    mask2,center=analysis_blob(mask)
    for i in range(10):
        cv2.circle(image, (int(center[i][0]),int(center[i][1])), 3, (255, 0, 0), -1)

    return image,center


def analysis_blob(mask):
    label = cv2.connectedComponentsWithStats(mask)
    data = np.delete(label[2], 0, 0)
    center = np.delete(label[3], 0, 0) 
    center2=np.zeros((10,2))
    if len(data)!=0:
        for i in range(10):
            max_index = np.argmax(data[:,4])
            center2[i]=center[max_index]
            data=np.delete(data,max_index,0)
            center=np.delete(center,max_index,0)
    mask=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    return mask,center2

file_path = 'movie_005.mp4'
delay = 1
window_name = 'frame'
csvfile='movie_005.csv'
cap = cv2.VideoCapture(file_path)
with open(csvfile,'w',newline="") as f:
    writer = csv.writer(f)

if not cap.isOpened():
    sys.exit()
count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
c=0
while True:
    ret, frame = cap.read()
    if ret:
        red,centerr=red_detect(frame)
        cv2.imshow(window_name,red)
        with open(csvfile,'a',newline="") as f:
            writer = csv.writer(f)
            writer.writerow([centerr[0][0],centerr[0][1],centerr[1][0],centerr[1][1],centerr[2][0],centerr[2][1],centerr[3][0],centerr[3][1],centerr[4][0],centerr[4][1],centerr[5][0],centerr[5][1],centerr[6][0],centerr[6][1],centerr[7][0],centerr[7][1],centerr[8][0],centerr[8][1],centerr[9][0],centerr[9][1]])
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    else:
        #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        break
    

cv2.destroyWindow(window_name)