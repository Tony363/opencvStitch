import numpy as np
import cv2 as cv
from utils import timer

img = cv.resize(cv.imread("left.png"),(480,640))
theta = np.radians(30)
c,s = np.cos(theta),np.sin(theta)
rmatrix = np.asarray([(c,-s,0),(s,c,0),(0,0,1)])
print(rmatrix)
print(img.shape)

for i,row in enumerate(img):
    for j,col in enumerate(row):
        pixel_data = img[i,j,:]
        input_coords = np.array([i,j,1])
        i_out,j_out,_ = rmatrix * input_coords
        start_time = timer(start_time=None)
        img[i_out.astype(np.int64),j_out.astype(np.int64),:] = pixel_data
        print(timer(start_time=start_time))
cv.imwrite("rotate.png",img)
