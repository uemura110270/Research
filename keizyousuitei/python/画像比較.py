# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 13:56:21 2020

@author: Shuya Matsuzaki
"""

import cv2
import os

TARGET_FILE = '05.png'
IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '/images/'
IMG_SIZE = (200, 200)

target_img_path = IMG_DIR + TARGET_FILE
target_img = cv2.imread(target_img_path)
target_img = cv2.resize(target_img, IMG_SIZE)
target_hist = cv2.calcHist([target_img], [0], None, [256], [0, 256])

print('TARGET_FILE: %s' % (TARGET_FILE))

files = os.listdir(IMG_DIR)
for file in files:
    if file == '.DS_Store' or file == TARGET_FILE:
        continue

    comparing_img_path = IMG_DIR + file
    comparing_img = cv2.imread(comparing_img_path)
    comparing_img = cv2.resize(comparing_img, IMG_SIZE)
    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])

    ret = cv2.compareHist(target_hist, comparing_hist, 0)
    print(file, ret)