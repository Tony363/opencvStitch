#!/usr/bin/env python

'''
Stitching sample
================

Show how to use Stitcher API from python in a simple way to stitch panoramas
or scans.
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import imutils
import argparse
import sys
import time



modes = (cv2.Stitcher_PANORAMA, cv2.Stitcher_SCANS)
out_path = "result.mp4"

class CODES:
    INFO = "[INFO]"
    ERROR = "[ERROR]"

# Return time elapsed
def timer(start_time=None):
    if start_time is None:
        return time.time()
    return time.time() - start_time

def status_check(status):
    if status != cv2.Stitcher_OK:
        print("Can't stitch images, error code = {}".format(status))
        return False
    return True

def view_stitch(pano):
    resized = imutils.resize(cv2.UMat.get(pano),width=1920)
    cv2.imshow('Stitched Panorama',resized)

def get_minimum_total_frame(left_capture, right_capture):
    left_total_frame = int(left_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    right_total_frame = int(right_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frame = min(left_total_frame, right_total_frame)
    print(CODES.INFO, "Total frames set to {}".format(total_frame))
    return total_frame

def read_vid(stitcher,videos,stop_frame = None,view=False):
    capL = cv2.VideoCapture('{videoL}'.format(videoL=videos[0]))
    capR = cv2.VideoCapture('{videoR}'.format(videoR=videos[1]))
    
    # Set a maximum number of frames to write in the output video object
    # By default, the smallest total frames count among the two videos is assigned to stop_frame
    if stop_frame is None:
        stop_frame = get_minimum_total_frame(capL,capR)


    # NB : cv2.UMat array is faster than np array
    pano = cv2.UMat(np.asarray([]))
    readFrame = 0
    execution_time = timer()

    while True:
        start_time = timer()
        Lret,Lframe = capL.read()
        Rret,Rframe = capR.read()
        
        if Lret and Rret:
            # Estimate the transform on first frame
            if readFrame == 0 :

                status = stitcher.estimateTransform([Lframe,Rframe])
                if status_check(status):
                    print(CODES.INFO, "Transform successfully estimated")
                
                status,pano = stitcher.composePanorama([Lframe,Rframe],pano)
                if not status_check(status):
                    continue

                # Initialize the video writer object
                h,w = cv2.UMat.get(pano).shape[:2] # Convert UMat to numpy array
                fps = min(capL.get(cv2.CAP_PROP_FPS),capR.get(cv2.CAP_PROP_FPS))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(out_path,fourcc,fps,(w,h))
                print(CODES.INFO, "Video writer initialized with {:.1f} fps and shape {}x{}".format(fps,w,h))
            
            # Stitch the two images
            else:
                if readFrame == stop_frame:
                    print(CODES.INFO,"Stitching stop frame reached.")
                    break

                status, pano = stitcher.composePanorama([Lframe,Rframe],pano)
                if not status_check(status):
                    print(CODES.ERROR, "composePanorama failed.")
                
            
            # View the stitched panorama. Press "q" to quit the program.
            if view:
                view_stitch(pano)
                if cv2.waitKey(1) == ord('q'):
                    print(CODES.INFO, "Successfully quit the program.")
                    break

            print(CODES.INFO, "{}/{} Stitched successfully. Done in {:.3f}s".format(readFrame, stop_frame,timer(start_time)))
            out.write(pano)
            readFrame += 1

        else:
            print(CODES.ERROR, "Can't read the images")
            sys.exit(-1) 


        
    # Display Execution time
    print(CODES.INFO, "Total execution time {:.2f}s".format(timer(execution_time)))
    # Clean memory
    print(CODES.INFO, "Clean memory ...")
    capL.release()
    capR.release()
    out.release()
    return None


def img_write(stitcher,image,output):
    # read input images 
    imgs = []
    for img_name in args.img:
        img = cv2.imread(cv2.samples.findFile(img_name))
        if img is None:
            print("Can't read image " + img_name)
            sys.exit(-1)
        imgs.append(img)
    status, pano = stitcher.stitch(imgs)
    cv2.imwrite(args.output, pano)
    print("stitching completed successfully. %s saved!" % args.output)
    print('Done')

def main(args):
    stitcher = cv2.Stitcher.create(args.mode)
    if args.videos:
        read_vid(stitcher,args.videos,args.stop_frame,args.view)
    if args.img and args.output:
        img_write(stitcher,args.img,args.output)

def command_args():
    parser = argparse.ArgumentParser(prog='stitching.py', description='Stitching sample.')
    parser.add_argument('--mode',type = int, choices = modes, default = cv2.Stitcher_PANORAMA,
    help = 'Determines configuration of stitcher. The default is `PANORAMA` (%d), '
            'mode suitable for creating photo panoramas. Option `SCANS` (%d) is suitable '
            'for stitching materials under affine transformation, such as scans.' % modes)
    parser.add_argument('--videos',nargs='+',help='input videos')
    parser.add_argument('--img', nargs='+', help = 'input images')
    parser.add_argument('--output', default = 'result.mp4',help = 'Resulting video. The default output name is `result.mp4`.')
    parser.add_argument('--stop_frame',type=int,help='Limit of frames to stitch')
    parser.add_argument('--view',action='store_true',help='view stitch in windows')
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

