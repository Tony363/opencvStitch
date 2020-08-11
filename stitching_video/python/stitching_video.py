#!/usr/bin/env python3

'''
Threaded stitching
================
Stitch videos from files or USB/MIPI cameras using different threads for reading the frames, stitching and displaying
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import imutils
import argparse
import sys


from camera import CSI_Camera,Panorama, get_minimum_total_frame, status_check
from gstreamer import gstreamer_pipeline
from utils import *

# Stitching variables
left_camera = None
right_camera = None
final_camera = None

left_image = None
right_image = None
pano = None

modes = (cv2.Stitcher_PANORAMA, cv2.Stitcher_SCANS)

SAVE = False
OUT_PATH = "result.mp4"
DONE = False


def view_stitch(pano):
    resized = imutils.resize(cv2.UMat.get(pano),width=1920)
    cv2.imshow('Stitched Panorama',resized)



# This read_vid function will display the left/right frames, the result panorama in the same thread
# BUT the stitching is done separetely in another thread
# Display 10 FPS 
def read_vid_thread(stitcher,interface,device0,device1,capture_width, capture_height,videos,stop_frame = None,view=False, display_width=1080):
    global left_camera, right_camera, left_image, right_image, final_camera, pano
    left_camera = CSI_Camera(interface, capture_width, capture_height)
    right_camera = CSI_Camera(interface, capture_width, capture_height)

    # Use offline videos file
    if interface=="none" and videos is not None:
        left_camera.open(interface,videos[0],capture_width, capture_height)
        right_camera.open(interface,videos[1],capture_width, capture_height)

    # Use the MIPI interface cameras
    elif interface=="mipi":
        
        left_camera.open(interface,gstreamer_pipeline(
            sensor_id=0,
            sensor_mode=3,
            flip_method=0,
            display_height=540,
            display_width=960,
        ),capture_width, capture_height)
        right_camera.open(interface,gstreamer_pipeline(
            sensor_id=1,
            sensor_mode=3,
            flip_method=0,
            display_height=540,
            display_width=960,
        ),capture_width, capture_height)
     # Use the USB interface cameras
    elif interface=="usb":
        left_camera.open(interface,device0,capture_width, capture_height)
        right_camera.open(interface,device1,capture_width, capture_height)
    else:
        print(CODES.ERROR,"Interface does not exist.")
        SystemExit(0)


    left_camera.start()
    right_camera.start()

    # Initialize Panorama class
    final_camera = Panorama(left_camera, right_camera,stop_frame,SAVE, OUT_PATH)
    final_camera.start()

    if (not left_camera.video_capture.isOpened()
        or not right_camera.video_capture.isOpened()):
        # Cameras did not open, or no camera attached
        print("Unable to open any cameras")
        SystemExit(0)


    while True:
        _ , left_image=left_camera.read()
        _ , right_image=right_camera.read()
        
        input_display_time = timer()
        camera_images = np.hstack((left_image, right_image)) #70 ms on 4K, 7ms on width 640
        camera_images = imutils.resize(camera_images, width = display_width)
        if view:
            cv2.imshow("Left/Right Cameras", camera_images)
        timer(input_display_time, "input_display_time")
        
        # Show the panorama stream (stitched video streams)
        _, pano = final_camera.read()
        pano_display_time = timer()
        if pano is not None and view:
            pano = imutils.resize(pano, width = display_width)
            cv2.imshow("Stitched view", pano)


            wait_key_time = timer()
            keyCode = cv2.waitKey(30) & 0xFF
            timer(wait_key_time,"wait_key_time")
            # Stop the program on the ESC key
            if keyCode == 27:
                print(CODES.INFO, "Successfully quit the program.")
                break

        timer(pano_display_time,"pano_display_time")

        # Properly quit the main Thread
        # if the stitching is done or has reached the set limit frames
        if final_camera.isDone:
            break

    cleanMemory()

    return None


# This read_vid function will display the left/right frames, stitch and display the result in the same thread
# Display is 2FPS (500ms)
def read_vid(stitcher,interface,capture_width,capture_height,videos,stop_frame = None,view=False):
    global left_camera
    global right_camera
    left_camera = CSI_Camera(interface, capture_width, capture_height)
    right_camera = CSI_Camera(interface, capture_width, capture_height)
    # Use offline videos file
    if interface=="none" and videos is not None:
        left_camera.open(interface,videos[0],capture_width, capture_height)
        right_camera.open(interface,videos[1],capture_width, capture_height)

    # Use the MIPI interface cameras
    elif interface=="mipi":
        
        left_camera.open(interface,gstreamer_pipeline(
            sensor_id=0,
            sensor_mode=3,
            flip_method=0,
            display_height=540,
            display_width=960,
        ),capture_width, capture_height)
        right_camera.open(interface,gstreamer_pipeline(
            sensor_id=1,
            sensor_mode=3,
            flip_method=0,
            display_height=540,
            display_width=960,
        ),capture_width, capture_height)
     # Use the USB interface cameras
    elif interface=="usb":
        left_camera.open(interface,0,capture_width, capture_height)
        right_camera.open(interface,1,capture_width, capture_height)
    else:
        print(CODES.ERROR,"Interface does not exist.")
        SystemExit(0)
    
    # Start cameras frame reading thread
    left_camera.start()
    right_camera.start()



    if (not left_camera.video_capture.isOpened()
        or not right_camera.video_capture.isOpened()):
        # Cameras did not open, or no camera attached
        print("Unable to open any cameras")
        SystemExit(0)


    # Set a maximum number of frames to write in the output video object
    # By default, the smallest total frames count among the two videos is assigned to stop_frame
    if stop_frame is None:
        stop_frame = get_minimum_total_frame(left_camera.video_capture,right_camera.video_capture)


    # NB : cv2.UMat array is faster than np array
    pano = cv2.UMat(np.asarray([]))
    readFrame = 0
    execution_time = timer()

    while True:
       
        start_time = timer()
        Lret , left_frame=left_camera.read()
        Rret , right_frame=right_camera.read()
        camera_images = np.hstack((left_frame, right_frame))
        camera_images = imutils.resize(camera_images, width = 1980)
        cv2.imshow("CSI Cameras", camera_images)

        if Lret and Rret:
            # Estimate the transform on first frame
            if readFrame == 0 :
                status = stitcher.estimateTransform([left_frame,right_frame])
                if status_check(status):
                    print(CODES.INFO, "Transform successfully estimated")
                
                status,pano = stitcher.composePanorama([left_frame,right_frame],pano)
                if not status_check(status):
                    continue

                # Initialize the video writer object
                capL = left_camera.video_capture
                capR = right_camera.video_capture
                h,w = cv2.UMat.get(pano).shape[:2] # Convert UMat to numpy array
                fps = min(capL.get(cv2.CAP_PROP_FPS),capR.get(cv2.CAP_PROP_FPS))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(OUT_PATH,fourcc,fps,(w,h))
                print(CODES.INFO, "Video Writer initialized with {:.1f} fps and shape {}x{}".format(fps,w,h))
            
            # Stitch the two images
            else:
                if readFrame == stop_frame:
                    print(CODES.INFO,"Stitching stop frame reached.")
                    break
                
                compose_time = timer()
                status, pano = stitcher.composePanorama([left_frame,right_frame],pano)
                timer(compose_time, "compose_time")

                if not status_check(status):
                    print(CODES.ERROR, "composePanorama failed.")

            
            # View the stitched panorama. Press "q" to quit the program.
            if view:
                view_stitch(pano)
                if cv2.waitKey(1) == ord('q'):
                    print(CODES.INFO, "Successfully quit the program.")
                    break

            print(CODES.INFO, "{}/{} Stitched successfully. Done in {:.3f}s".format(readFrame, stop_frame,timer(start_time)))
            if SAVE:
                out.write(pano)
            readFrame += 1

        else:
            print(CODES.ERROR, "Can't read the images")
            sys.exit(-1) 


        
    # Display Execution time
    timer(execution_time,"execution_time")
    # Clean memory
    print(CODES.INFO, "Clean memory ...")
    left_camera.stop()
    left_camera.release()
    right_camera.stop()
    right_camera.release()

    out.release()

    return None

def cleanMemory():
    left_camera.stop()
    left_camera.release()
    right_camera.stop()
    right_camera.release()
    final_camera.stop()
    cv2.destroyAllWindows()
    print(CODES.INFO, "Memory cleaned successfully")

def main(args):
    stitcher = cv2.Stitcher.create(args.mode)

    SAVE = args.save
    OUT_PATH = args.output
    
    read_vid_thread(stitcher,args.interface,args.device0,args.device1,args.capture_width,args.capture_height,args.videos,args.stop_frame,args.view, args.display_width)
    

# Python 3 
# ex usage (local) : python3 stitching_video.py --interface none --videos ../inputs/left.mp4 ../inputs/right.mp4 --view --capture_width 640 --capture_height=480
# ex usage (USB camera) : python3 stitching_video.py --device0 0 --device1 1 --capture_width 640 --capture_height=480 --view 

def command_args():
    parser = argparse.ArgumentParser(prog='stitching_video.py', description='threaded stitching sample.')
    parser.add_argument('--mode',type = int, choices = modes, default = cv2.Stitcher_PANORAMA,
    help = 'Determines configuration of stitcher. The default is `PANORAMA` (%d), '
            'mode suitable for creating photo panoramas. Option `SCANS` (%d) is suitable '
            'for stitching materials under affine transformation, such as scans.' % modes)
    parser.add_argument('--interface', default='usb',help='define the cameras interface (usb|mipi|none)')
    parser.add_argument('--device0', type=int, default=0,help='Left camera device id')
    parser.add_argument('--device1', type=int, default=1,help='Right camera device id')
    parser.add_argument('--capture_width', type=int, help='Cameras capture width')
    parser.add_argument('--capture_height', type=int, help='Cameras capture height')
    parser.add_argument('--videos',nargs='+',help='input videos. To use videos file, set \'interface\' to none')
    parser.add_argument('--save', action='store_true',help = 'Save the stitched result')
    parser.add_argument('--output', default = 'result.mp4',help = 'Resulting video. The default output name is `result.mp4`.')
    parser.add_argument('--stop_frame',type=int,help='Limit of frames to stitch')
    parser.add_argument('--view',action='store_true',help='view stitch in windows')
    parser.add_argument('--display_width', type=int,default=1080, help='Cameras display width')
    args = parser.parse_args()
    return parser,args

if __name__ == '__main__':
    parser,args = command_args()
    """
    __doc__ += '\n' + parser.format_help()
    print(__doc__)
    """
    main(args)
    cv2.destroyAllWindows()

