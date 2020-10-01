import numpy as np
import cv2 as cv
from utils import timer

img = cv.resize(cv.imread("inputs/left.png"),(640,480))
theta = np.radians(30)
c,s = np.cos(theta),np.sin(theta)
rmatrix = np.asarray(
    [
    (c,s,0),
    (-s,c,0),
    (0,0,1)
        ])
T_s = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
T_r = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
T = T_s @ T_r
img_transform = np.empty((480*2,640*2,3),dtype=np.uint8)
for i,row in enumerate(img):
    for j,col in enumerate(row):
        pixel_data = img[i,j,:]
        input_coords = np.array([i,j,1])
        i_out,j_out,_ =  rmatrix @ input_coords
        start_time = timer(start_time=None)
        img_transform[i_out.astype(np.int64),j_out.astype(np.int64),:] = pixel_data
        print(timer(start_time=start_time))
cv.imwrite("outputs/rotate.png",img_transform)
