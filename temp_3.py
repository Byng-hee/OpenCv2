import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('soccer.jpg')
# [1] 영상이미지는 리스트로 주어야함, [2] 2번 채널인 R의 히스토그램을 구한다.
# [3] None부분은 마스크를 나타내는데 Noned이라서 전부분, [4]256칸을 보여주라는 뜻
# [5] 명암을 나타낸 부분이다.
h = cv.calcHist([img],[0],None,[256],[0,256])
plt.plot(h, color='r', linewidth=1)