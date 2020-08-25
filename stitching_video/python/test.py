import cv2
import imutils
from UMatFileVideoStream import UMatFileVideoStream

Lvideo = UMatFileVideoStream('left.mp4', 128).start()
Rvideo = UMatFileVideoStream('right.mpm4',128).start()
rgb = cv2.UMat(720,1080, cv2.CV_8UC3)
# video.update()
while not (Lvideo.stopped and Rvideo.stopped):
    Lret,Rret = Lvideo.more(),Rvideo.more()
    Lframe,Rframe = Lvideo.read(),Rvideo.read()
    print(Lframe,Rframe)
    # Lresized = imutils.resize(Lframe,width=1080)
    # Rresized = imutils.resize(Rframe,width=1080)
    cv2.imshow('test',Lframe)
    cv2.imshow('test1',Rframe)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        Lvideo.stop()
        Rvideo.stop()
   

    
    
