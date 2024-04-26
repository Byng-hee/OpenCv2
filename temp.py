import cv2 as cv
import sys

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit("그딴거 없다.")
    
    
BrushSize=5
LColor, RColor = (255,0,0), (0,0,255)    

brush_box_x, brush_box_y, brush_box_w, brush_box_h = 10, 10, 100, 30

def painting(event, x, y, flags, param):
    
    global BrushSize
    
    if (event==cv.EVENT_LBUTTONDOWN or event==cv.EVENT_RBUTTONDOWN) and (brush_box_x<=x<=brush_box_x+brush_box_w and brush_box_y<=y<=brush_box_y+brush_box_h):
        BrushSize += 1
    elif event==cv.EVENT_LBUTTONDOWN:
        cv.circle(img, (x,y), BrushSize,LColor, -1)
    elif event==cv.EVENT_RBUTTONDOWN:
        cv.circle(img, (x,y), BrushSize, RColor, -1)
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_LBUTTON:
        cv.circle(img, (x,y), BrushSize, LColor, -1)
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_RBUTTON:
        cv.circle(img, (x,y), BrushSize, RColor, -1)
    
    cv.imshow('Drawing',img)
        

cv.namedWindow('Drawing')
cv.rectangle(img, (brush_box_x, brush_box_y), (brush_box_x + brush_box_w, brush_box_y + brush_box_h), (255, 255, 255), -1)
cv.putText(img, 'Brush Size', (brush_box_x + 5, brush_box_y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

cv.imshow('Drawing', img)

cv.setMouseCallback('Drawing', painting)

while(True):
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break