import cv2 as cv
import sys
import os
import shutil



#img = cv.imread('soccer.jpg')
img = cv.imread('girl_laughing.jpg')
if img is None:
   sys.exit('파일을 찾을 수 없습니다.')
   
# BrushSize = 5
# LColor, RColor = (255,0,0),(0,0,255)

#def painting(event, x,y, flags, param):
#    if event == cv.EVENT_LBUTTONDOWN:
#        cv.circle(img, (x,y), BrushSize, LColor, -1)
#    elif event==cv.EVENT_RBUTTONDOWN:
#        cv.circle(img, (x,y), BrushSize, RColor, -1)
#    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
#        cv.circle(img,(x,y), BrushSize, LColor, -1)
#    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
#        cv.circle(img, (x,y), BrushSize, RColor, -1)
#        
#    cv.imshow('Painting', img)


def draw(event,x,y,flags,param):
    if event==cv.EVENT_LBUTTONDOWN:
        cv.rectangle(img,(x,y),(x+200,y+200),(0,0,255),2)
    elif event==cv.EVENT_RBUTTONDOWN:
        cv.rectangle(img,(x,y),(x+100,y+100),(255,0,0),2)
        
    cv.imshow('Drawing',img)

#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#gray_small = cv.resize(gray, dsize=(0,0), fx=0.5, fy=0.5)

#cv.imwrite('sccer_gray.jpg', gray)
#cv.imwrite('sccer_gray_small.jpg', gray_small)

#cv.imshow('Gray image',gray)
#cv.imshow('Gray image small', gray_small)

# cv.namedWindow('Painting')
# cv.imshow('Painting', img)
#
# cv.setMouseCallback('Painting', painting)

cv.namedWindow('Drawing')
cv.imshow('Drawing', img)

cv.setMouseCallback('Drawing', draw)

while(True):
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break


# cv.rectangle(img,(830,30),(1000,200),(0,0,255),2)
# cv.putText(img,'laugh',(830,24),cv.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
# 
# cv.imshow('Draw',img)

# cv.waitKey()
# cv.destroyAllWindows()              
    
    


    
    