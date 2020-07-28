import cv2
import threading
import time
import sys
import imutils


class CSI_Camera:

    def __init__ (self) :
        # Initialize instance variables
        # OpenCV video capture element
        self.video_capture = None
        # The last captured image from the camera
        self.frame = None
        self.grabbed = False
        # The thread where the video capture runs
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

    # Open CSI-cameras with GStreamer
    # for mipi, filename is the gstreamer pipeline string returned by gstreamer.py
    # for usb, filename is the camera device ID (0 or 1)
    def open(self, interface, filename):
        if interface  == "mipi":
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

        elif interface == "usb" or interface == "none":
            try:
                self.video_capture = cv2.VideoCapture(filename) 
                print("{} Camera {} successfully opened".format(interface,filename))
            except RuntimeError:
                self.video_capture = None
                print("Unable to open camera")
                print("Pipeline: " + filename)
                return
            # Grab the first frame to start the video capturing
            self.grabbed, self.frame = self.video_capture.read()
            
    """
    # Open videos file using opencv VideoCapture
    def openFile(self, interface, video_name):
        try:
            self.video_capture = cv2.VideoCapture(video_name)
            
        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            return
        # Grab the first frame to start the video capturing
        self.grabbed, self.frame = self.video_capture.read()
    """

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

    def updateCamera(self):
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                if grabbed:
                    with self.read_lock:
                        self.grabbed=grabbed
                        self.frame=frame
            except RuntimeError:
                print("Could not read image from camera")
        # FIX ME - stop and cleanup thread
        # Something bad happened
        

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
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
    def __init__(self, left_camera, right_camera):
        # panorama image
        self.status = None
        self.pano = None
        # Initialize Stitcher class
        self.stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        self.imgs = []
        self.stiched_frames = 0

        # Initialize CSI cameras
        self.left_camera = left_camera
        self.right_camera = right_camera


        # The thread where the video capture runs
        self.read_thread = None
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

    def stitchCamera(self):

            # This is the thread to read images from the camera
            while self.running:
                try:
                    # Initialize left and right frames
                    # CSI cameras frames works on 30 or 60 FPS but the sticher works under 3FPS (slower)
                    # Therefore it needs to store a frame for a longer period of time to be able to stitch
                    if self.left_camera.frame is not None and self.right_camera.frame is not None:
                        _, left_image = self.left_camera.read()
                        _, right_image = self.right_camera.read()

                    stitch_start_time = time.time()
                    
                    status,pano = self.stitcher.stitch([left_image, right_image])

                    if status != cv2.Stitcher_OK:
                        print("Can't stitch images, error code = %d" % status)
                        #sys.exit(-1) 

                    print("Stitching completed successfully ({}). Done in {:.3f}s".format(self.stiched_frames,time.time() - stitch_start_time))

                    with self.read_lock:
                        self.status=status
                        self.pano=pano
                        self.stiched_frames += 1

                except RuntimeError:
                    print("Could not stitch image from CSI cameras")
            # FIX ME - stop and cleanup thread
            # Something bad happened

    def read(self):
        if self.pano is not None:
            with self.read_lock:
                pano = self.pano.copy()
                status = self.status
            return status, pano 
        else:
            return None, None


    def stop(self):
        self.running=False
        self.read_thread.join()
