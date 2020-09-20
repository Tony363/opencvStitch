from __future__ import print_function
# from Stitcher_class import Manual
# from utils import Manual_Detailed,Kseam_work_aspect
from detail_stitch import Manual_Detailed,Kseam_work_aspect
from imutils.video import VideoStream
import numpy as np
import datetime
import imutils
import time
import cv2 as cv

leftStream = cv.VideoCapture('inputs/left.mp4') 
rightStream = cv.VideoCapture('inputs/right.mp4')

# initialize the image stitcher, motion detector, and total
# number of frames read
# stitcher = Manual()
total = 0

# loop over frames from the video streams
while True:
    # grab the frames from their respective video streams
    ret,left = leftStream.read()
    ret,right = rightStream.read()
    
    # stitch the frames together to form the panorama
    # IMPORTANT: you might have to change this line of code
    # depending on how your cameras are oriented; frames
    # should be supplied in left-to-right order
    # status,result = stitcher.stitch([left, right])
    status,pano,cached = Manual(left,right)

    # left = imutils.resize(left, width=1080)
    # right = imutils.resize(right, width=1080)
    result = imutils.resize(pano,width=1080)
    # no homograpy could be computed
    if result is None:
        print("[INFO] homography could not be computed")
        break

    # increment the total number of frames read and draw the 
    # timestamp on the image
    total += 1
    timestamp = datetime.datetime.now()
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    # cv2.putText(result, ts, (10, result.shape[0] - 10),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    # show the output images
    cv.imshow("Result", result)
    # cv.imshow("Left Frame", left)
    # cv.imshow("Right Frame", right)
    key = cv.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
print("[INFO] cleaning up...")
cv.destroyAllWindows()
leftStream.release()
rightStream.release()