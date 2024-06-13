'''
총 4 문제 : 2문제는 응용 2문제는 기본형태
4-1, 4-3, 4-4, 4-5, 4-6, [4-7], 5-2, [5-3] // []은 확정임

[4-7]은 응용문제
사용자가 붓칠을 하면 분할 1번하고 종료됨 마음에 들지 않으면 사용자 추가 붓칠 프로그램은 이를 갱신해주는 것!!
40~43번 line을 함수화 하고 나머지 손대면 된다.
#---------------------------------------------------------------------------4-1
#소벨 엣지 검출 : 엣지의 윤곽을 표현한다. 색깔의 변화가 심한곳을 경계로 나눈다.
import cv2 as cv

img = cv.imread('soccer.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

grad_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize = 3)
grad_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize = 3)

sobel_x = cv.convertScaleAbs(grad_x)
sobel_y = cv.convertScaleAbs(grad_y)

edge_strength = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

cv.imshow('Original', gray)
cv.imshow('sobelx', sobel_x)
cv.imshow('sobely', sobel_y)
cv.imshow('edge strength', edge_strength)

cv.waitKey()
cv.destroyAllWindows()


#---------------------------------------------------------------------------4-3
#엣지 맵에서 경계선을 찾는것이다. 엣지 맵에서 sobel보다 한층 발전한 그림자까지 구분이 되는 canny를 
#사용해서 경계선을 찾는 것이다. 결국 엣지 추출
import cv2 as cv
import numpy as np

img = cv.imread('soccer.jpg')   #영상 읽기
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
canny = cv.Canny(gray, 100, 200)    #T-low = 50, T-high = 150 으로 윤곽의 정도에 따라 레벨 수준의 범위 결정

#이미 윤곽을 canny로 거른 상태. 즉, edge map이 형성된 상태. 윤곽 검출 모드와 윤곽근사화방법을 사용한다.
contour, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
#contour = 윤곽선 딴거의 list로 저장, hierarchy = 윤곽선의 계층 정보를 저장한 리스트

lcontour = []   #contour중에서 100이상인 윤곽선들을 저장할 것!
for i in range(len(contour)):
    if contour[i].shape[0] > 100:  #정확히 말하면 윤곽의 길이가 100이상인 경우에 저장
        lcontour.append(contour[i])
        
cv.drawContours(img, lcontour, -1, (0,255,0), 3)    #img에 lcontour의 윤곽선들을  G색으로 
#3굵기로 그린다. 또한 -1 즉 pos 3은 모든 윤곽을 그리라~ 라는 말이다.


cv.imshow('Original with contours', img)
cv.imshow('Canny', canny)

cv.waitKey()
cv.destroyAllWindows()


#---------------------------------------------------------------------------4-4
#허프만 연산 실시 - 원과 직선으로 물체 윤곽 표시
import cv2 as cv

img = cv.imread('apples.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

apples = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 200, param1=150, param2=20,
                         minRadius=50, maxRadius = 120)
# gray 이미지에 대해서, 연산 수행, 해상도는 1, 원들의 중심점의 최소 간격 200, 캐니엣지의 임계값, param2는 작을수록 원이 많음

for i in apples[0]:
    cv.circle(img, ( int(i[0]), int(i[1])), int(i[2]), (255,0,0), 2)
    
cv.imshow('Apple detection', img)

cv.waitKey()
cv.destroyAllWindows()


#---------------------------------------------------------------------------4-5
#conda install scikit-image 설치 conda에

import skimage
import numpy as np
import cv2 as cv

img = skimage.data.coffee()
cv.imshow('Coffee imgae', cv.cvtColor(img, cv.COLOR_RGB2BGR))

# 600개의 segmentaion의 갯수로 나누어서 20의 조밀함을 가지는 slic을 생성한다// 인수들 모두 기억해야한다. 반드시 작성 필수
slic1 = skimage.segmentation.slic(img, compactness = 20, n_segments = 600)
sp_img1 = skimage.segmentation.mark_boundaries(img, slic1)
sp_img1 = np.uint8(sp_img1*255.0)

slic2 = skimage.segmentation.slic(img, compactness = 40, n_segments = 600)
marking = skimage.segmentation.mark_boundaries(img, slic2)
sp_img2 = np.uint8(sp_img2*255.0)

cv.imshow('Super pixels (compact 20)', cv.cvtColor(sp_img1, cv.COLOR_RGB2BGR))
cv.imshow('Super pixels (compact 40)', cv.cvtColor(sp_img2, cv.COLOR_RGB2BGR))

cv.waitKey()
cv.destroyAllWindows()


#---------------------------------------------------------------------------4-6
#정규화 분할 skimage의 slic을 이용한 색깔기반의 측정법으로
import skimage
import numpy as np
import cv2 as cv
import time

coffee = skimage.data.coffee()

start = time.time()
slic = skimage.segmentation.slic(coffee, compactness=20, n_segments=600, start_label=1)
g = skimage.graph.rag_mean_color(coffee, slic, mode = 'similarity')
ncut = skimage.graph.cut_normalized(slic, g) #정규화 절단
print(coffee.shape, 'Coffee 영상을 분할하는 데', time.time()-start, '초 소요')

marking = skimage.segmentation.mark_boundaries(coffee, ncut)
ncut_coffee = np.uint8(marking*255.0)

cv.imshow('Normalized cut', cv.cvtColor(ncut_coffee, cv.COLOR_RGB2BGR))

cv.waitKey()
cv.destroyAllWindows()


#---------------------------------------------------------------------------4-7
import cv2 as cv
import numpy as np

img = cv.imread('soccer.jpg')
img_show = np.copy(img)

mask = np.zeros( (img.shape[0], img.shape[1]), np.uint8)
mask[:, :] = cv.GC_PR_BGD

BrushSize = 9
LColor, RColor = (255,0,0), (0,0,255)

def painting(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img_show, (x,y), BrushSize, LColor, -1)
        cv.circle(mask, (x,y), BrushSize, cv.GC_FGD, -1)
    elif event == cv.EVENT_RBUTTONDOWN:
        cv.circle(img_show, (x,y), BrushSize, RColor, -1)
        cv.circle(mask, (x,y), BrushSize, cv.GC_BGD, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img_show, (x,y), BrushSize, LColor, -1)
        cv.circle(mask, (x,y), BrushSize, cv.GC_FGD, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
        cv.circle(img_show, (x,y), BrushSize, RColor, -1)
        cv.circle(mask, (x,y), BrushSize, cv.GC_BGD, -1)
    
    cv.imshow('Painting', img_show)
    
cv.namedWindow('Painting')
cv.setMouseCallback('Painting', painting)

#여기부터 GrabCut적용하는 코드
background = np.zeros( (1,65), np.float64)
foreground = np.zeros( (1,65), np.float64)

def apply_grabcut():
    cv.grabCut(img, mask, None, background, foreground, 5,cv.GC_INIT_WITH_MASK)
    mask2 = np.where( (mask==cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
    grab = img*mask2[:,:,np.newaxis]
    
    cv.imshow('Grab cut image', grab)

while True:
    key = cv.waitKey(1)
    if key == ord('q'):  # 'q'를 누르면 종료
        break
    elif key == ord('g'):  # 'g'를 누르면 GrabCut 적용
        apply_grabcut()
    elif key == ord('r'):  # 'r'를 누르면 다시 그리기
        img_show = np.copy(img)
        mask = np.zeros( (img.shape[0], img.shape[1]), np.uint8)
        mask[:, :] = cv.GC_PR_BGD
        cv.namedWindow('Painting')
        cv.setMouseCallback('Painting', painting)

cv.waitKey()
cv.destroyAllWindows()


#---------------------------------------------------------------------------5-2
# sift 기술자를 통해서 특징을 추출할 수 있다.
import cv2 as cv

img = cv.imread('mot_color70.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)

gray = cv.drawKeypoints(gray, kp, None, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('sift', gray)

k = cv.waitKey()
cv.destroyAllWindows()


#---------------------------------------------------------------------------5-3
'''
import cv2 as cv
import numpy as np
import time

img1 = cv.imread('mot_color70.jpg')[190:350, 440:560]
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('mot_color83.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
print('특징점 개수:', len(kp1), len(kp2) )

start = time.time()
flann_matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
knn_match = flann_matcher.knnMatch(des1, des2, 2)

T = 0.7
good_match = []
for nearest1, nearest2 in knn_match:
    if(nearest1.distance/nearest2.distance) < T:
        good_match.append(nearest1)
print('매칭에 걸린 시간: ', time.time()-start)

img_match = np.empty(
                (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3),
                dtype=np.uint8
            )

cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('Good Matches', img_match)
k=cv.waitKey()
cv.destroyAllWindows()                   
'''