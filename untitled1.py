import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('mistyroad.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray'), plt.xticks([]), plt.yticks([]), pltshow()

h = cv.calcHist([gray], [0], None, [256], [0,256])
plt.plot(h, colot='r', linewidth=1), plt.show()

equal=cv.equalizeHist(gray)
plt.imshow(equal, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

h = cv.calcHist([equal],[0], None, [256], [0,256])
plt.plot(h, color='r', linewidth=1), plt.show()

"""
import cv2 as cv
import numpy as np

img = cv.imread('soccer.jpg')
img = cv.resize(img, dsize=(0,0), ty=0.4, fy=0.4)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.putText(gray, 'soccer', (10,,20), cv.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
cv.imshow('Original',gray)

smooth=np.hstack((cv.GaussianBlur(gray, (5,5), 0.0), cv.GaussianBlur(gray,(9,9),0.0),cv.GaussianBlur(gray,(15,15),0.0)))
cv.imshow('Smooth', smooth)

femboss = np.array([[-1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0]])

gray16 = np.int16(gray)
emboss = np.uin8(np.clip(cv.filter2D(gray16,-1,femboss)+128, 0, 255))
emboss_bad = np.uin8(cv.filter2D(gray16,-1,femboss)+128)
emboss_worse = cv.filter2D(gray16,-1,femboss)

cv.imshow('Emboss', emboss)
cv.imshow('Emboss', emboss_bad)
cv.imshow('Emboss', emboss_worse)

cv.waitKey()
cv.destroyAllWindows()
"""