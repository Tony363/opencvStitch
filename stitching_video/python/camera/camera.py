import cv2
import threading
import time
import sys
import imutils
import numpy as np
import collections

from os import path
sys.path.append(path.dirname(path.dirname(__file__)))

from math import ceil
from utils import *
from stitching_object import Stitcher

class CSI_Camera:

    def __init__ (self,interface,capture_width, capture_height) :
        # Initialize instance variables
        # Camera properties
        self.interface = interface
        self.capture_width = capture_width
        self.capture_height = capture_height
        # OpenCV video capture element
        self.video_capture = None
        # The last captured image from the camera
        self.frame = None
        self.grabbed = False
        # The thread where the video capture runs
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False
        # Openv-python GPU object
        # self.GPU = cv2.cuda_GpuMat()

    # Open CSI-cameras with GStreamer
    # for mipi, filename is the gstreamer pipeline string returned by gstreamer.py
    # for usb, filename is the camera device ID (0 or 1)
    def open(self, interface, filename, capture_width, capture_height,selectionRate=128):
        if interface  == "mipi":
            print("mipi",'\n')
            try:
                self.video_capture = cv2.VideoCapture(
                    filename, cv2.CAP_GSTREAMER
                ) # Use API CAP_GSTREAMER
                print("{} Camera {} successfully opened".format(interface,filename))

            except RuntimeError:
                self.video_capture = None
                print("Unable to open camera")
                print("Pipeline: " + filename)
                return
            # Grab the first frame to start the video capturing
            self.grabbed, self.frame = self.video_capture.read()
            # self.GPU.upload(self.frame)

        elif interface == "usb" or interface == "none":
            print("None",'\n')
            try:
                if interface == "none":
                    self.video_capture = cv2.VideoCapture(filename)
                elif interface == "usb":
                    self.video_capture = cv2.VideoCapture(filename,cv2.CAP_V4L) 
                print(CODES.INFO,"{} type Camera {} successfully opened".format(interface,filename))
                #https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
                if (capture_width is not None and capture_height is not None):
                    self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(capture_width)) # Set width of the frame in the video frame
                    self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(capture_height))
                
                print(CODES.INFO,"Capture width and height set to : {}x{}".format(
                    self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH),self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                # Video decoder (Speed performance)
                self.video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                print(CODES.INFO,"Video decoder set to : MJPG")

            except RuntimeError:
                self.video_capture = None
                print("Unable to open camera")
                print("Pipeline: " + filename)
                return

            # Grab the first frame to start the video capturing
            self.grabbed,self.frame = self.video_capture.read()
            # self.GPU.upload(self.frame)
            # If the video is a file (interface == "none") the video must be manually resized before stitch
            if self.interface == "none" and self.grabbed:
                self.frame = imutils.resize(self.frame, self.capture_width)
                # self.GPU.upload(self.frame)
            
    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        # create a thread to read the camera image
        if self.video_capture != None:
            self.running=True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running=False
        self.read_thread.join()
        print(CODES.INFO, "Video capturing thread quit.")

    def updateCamera(self):
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                # self.GPU.upload(frame)
                if grabbed:
                    # If the video is a file (interface == "none") the video must be manually resized before stitch
                    if self.interface == "none":
                        # frame = imutils.resize(self.GPU.download(), self.capture_width)
                        frame = imutils.resize(frame,self.capture_width)
                    # else:
                    #     frame = self.GPU.download()                 
                    with self.read_lock:
                        self.grabbed=grabbed
                        self.frame=frame
            except RuntimeError:
                print("Could not read image from camera")
        # FIX ME - stop and cleanup thread
        # Something bad happened
        

    def read(self):
        with self.read_lock:
            frame=self.frame.copy()
            grabbed=self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Now kill the thread
        if self.read_thread != None:
            self.read_thread.join()



class Panorama:
    def __init__(self, left_camera, right_camera,stop_frame, save, out_path,timer):
        # panorama image
        self.status = None
        self.pano = None
        self.save = save
        self.out = None # Video writer
        self.out_path = out_path
        self.fps_array = collections.deque(maxlen=5) # Store processing time for last 5 frames to estimate video writer FPS

        # batched saving
        self.memory_store = np.asarray([])
        # self.GPU = cv2.cuda_GpuMat()

        # Initialize Stitcher class
        self.stitcher = Stitcher
        self.to_estimate = None
        self.stitched_frames = 0
        self.timer = timer

        # Initialize CSI cameras
        self.left_camera = left_camera
        self.right_camera = right_camera

        # Limit of stitched frames
        self.isDone = False
        if (stop_frame is None and self.left_camera.interface == "none"):
            self.stop_frame = get_minimum_total_frame(left_camera.video_capture,right_camera.video_capture)
        else:
            self.stop_frame = stop_frame
        
        # The thread where the video stitching runs
        self.stitch_thread = None
        self.read_lock = threading.Lock()
        self.running = False
        self.save_thread = None

    def start(self):
        if self.running:
            print('Video sticher is already running')
            return None
        # create a thread to stitch the video streams
  
        self.running=True
        self.read_thread = threading.Thread(target=self.stitchCamera)
        self.read_thread.start()
        return self

    # Thread that stitches frames
    def stitchCamera(self):
            # NB : cv2.UMat array is faster than np array
            readFrame = 0
            while self.running:
                try:
                    # Initialize left and right frames
                    # CSI cameras frames works on 30 or 60 FPS but the sticher works under 3FPS (slower)
                    # Therefore it needs to store a frame for a longer period of time to be able to stitch
                    # if (self.left_camera.frame is not None and self.right_camera.frame is not None) or self.GPU:
                    if (self.left_camera.frame is not None and self.right_camera.frame is not None):
                        _, left_image = self.left_camera.read()
                        _, right_image = self.right_camera.read()

                        stitch_start_time = time.time()
                    
                        if self.stitched_frames == 0 or self.to_estimate is True:
                            # self.GPU.upload(pano)
                            for work_megapix in np.arange(0.01,1.0,0.01):
                                try:
                                    status,pano,stitcher = estimateTransform(left_image,right_image,work_megapix=0.6)
                                    self.stitcher = stitcher
                                    break
                                    # if pano.shape[0] in range(2000,2300) and pano.shape[1] in range(4700,5000):
                                    #     break
                                except Exception as e:
                                    print(e)
                                    continue
                            if status_check(status):
                                print(CODES.INFO, "Transform successfully estimated")
                                self.to_estimate = False
                            if not status_check(status):
                                continue

                            # Initialize the video writer object
                            if self.save:
                                capL = self.left_camera.video_capture
                                capR = self.right_camera.video_capture
                                # h,w = self.GPU.download().shape[:2]
                                h,w = pano.shape[:2] # Convert UMat to numpy array
                                fps = min(capL.get(cv2.CAP_PROP_FPS),capR.get(cv2.CAP_PROP_FPS))
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                
                                output_path = "outputs/"
                                timestamp = time.strftime('%b-%d-%Y_%H%M%S', time.localtime())
                                timestamp += ".mp4"
                                output_path+= timestamp
                                self.out = cv2.VideoWriter(output_path,fourcc,fps,(w,h))
                                print(CODES.INFO, "Video Writer initialized with {:.1f} fps and shape {}x{}".format(fps,w,h))
                            else:
                                print(CODES.INFO, "Initial left/right frame shape : {}x{}".format(left_image.shape[1],left_image.shape[0]))
                        
                        else:
                            # Quit stitching if the frame limit is reached
                            if readFrame == self.stop_frame:
                                self.isDone = True
                                print(CODES.INFO,"Stitching stop frame reached.")
                                break

                            compose_time = timer()
                            self.stitcher.new_frame(left_image,right_image)
                            pano = self.stitcher.composePanorama()
                            # self.GPU.upload(pano)
                            timer(compose_time, "compose_time", self.timer)
                            if not status_check(status):
                                print(CODES.ERROR, "composePanorama failed.") 
                                continue
                            
                        
                        excution_time = timer(stitch_start_time)
                        print(CODES.INFO,"Stitching completed successfully ({}/{}). Done in {:.3f}s. {}/{}".format(self.stitched_frames + 1,self.stop_frame,excution_time, "STORED" if self.save else "",self.memory_store.size if self.save else ""))
                        
                        self.fps_array.append(1/excution_time)
                        readFrame += 1
                        if self.save and self.out is not None:
                            self.memory_store = np.append(self.memory_store,cv2.UMat(pano))
                          
                        with self.read_lock:
                            self.status=status
                            # self.pano=self.GPU.download()
                            self.pano=pano
                            self.stitched_frames += 1

                except RuntimeError:
                    print("Could not stitch image from CSI cameras")
            # FIX ME - stop and cleanup thread
            # Something bad happened

    def read(self):
        if self.pano is not None:
            with self.read_lock:
                # pano = self.GPU.download().copy()
                pano = self.pano
                status = self.status
            return status, pano 
        else:
            return None, None


    def stop(self):
        self.running=False
        if self.out is not None:
            self.out.release()
        self.read_thread.join()
        print(CODES.INFO, "Video stitching thread quit.")




