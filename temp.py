import cv2 as cv
import sys

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
    
BrushSize = 5
LColor, RColor = (255,0,0),(0,0,255)

def painting(event, x,y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img, (x,y), BrushSize, LColor, -1)
    elif event==cv.EVENT_RBUTTONDOWN:
        cv.circle(img, (x,y), BrushSize, RColor, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img,(x,y), BrushSize, LColor, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
        cv.circle(img, (x,y), BrushSize, RColor, -1)
        
    cv.imshow('Painting', img)


#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#gray_small = cv.resize(gray, dsize=(0,0), fx=0.5, fy=0.5)

#cv.imwrite('sccer_gray.jpg', gray)
#cv.imwrite('sccer_gray_small.jpg', gray_small)

#cv.imshow('Gray image',gray)
#cv.imshow('Gray image small', gray_small)

cv.namedWindow('Painting')
cv.imshow('Painting', img)

cv.setMouseCallback('Painting', painting)

while(True):
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break