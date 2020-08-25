import cv2
import imutils
from UMatFileVideoStream import UMatFileVideoStream

Lvideo = UMatFileVideoStream('left.mp4', 1000).start()
Rvideo = UMatFileVideoStream('right.mpm4',1000).start()
rgb = cv2.UMat(720,1080, cv2.CV_8UC3)
# video.update()
while not (Lvideo.stopped and Rvideo.stopped):
    Lret,Rret = Lvideo.more(),Rvideo.more()
    Lframe,Rframe = Lvideo.read(),Rvideo.read()
    # print(ret,frame)
    Lresized = imutils.resize(Lframe,width=1080)
    Rresized = imutils.resize(Rframe,width=1080)
    cv2.imshow('test',Lresized)
    cv2.imshow('test1',Rresized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   

    
    
