import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('mistyroad.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
plt.imshow(gray, cmap = 'gray'), plt.xticks([]), plt.yticks([]), plt.show()

h = cv.calcHist([gray], [0], None, [256], [0,256])
plt.plot(h, color='r',linewidth=1), plt.show()

equal = cv.equalizeHist(gray)
plt.imshow(equal, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

h = cv.calcHist([equal], [0], None, [256], [0,256])
plt.plot(h, color='r',linewidth=1), plt.show()

"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('No file')
    
t, bin_img = cv.threshold(img[:,:,3], 0,255, cv.THRESH_BINARY + cv.THRESH_OTSU)
plt.imshow(bin_img, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

b = bin_img[bin_img.shape[0]//2:bin_img.shape[0], 0:bin_img.shape[0]//2+1]
plt.imshow(b, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

se= np.uint8([[0,0,1,0,0],
           [0,1,1,1,0],
           [1,1,1,1,1],
           [0,1,1,1,0],
           [0,0,1,0,0]])

dilation = cv.dilate(b,se, iteration=1)
plt.imshow(dilation, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

erosion = cv.erode(b, se, iteration=1)
plt.imshow(erosion, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

b_closing = cv.erode(cv.dilate(b,se, iteration=1), se, iteration=1)
plt.imshow(b_closing, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()
"""