#!/usr/bin/env python3


'''
Stream stitching
================

Stream the stitching done by stitching_video.py using a Flask server.
Left, right and stitched videos are displayed as preview and the user can start recording(saving)
'''


import cv2
import time
import threading
from flask import Response, Flask, render_template,redirect, request, url_for
from flask_fontawesome import FontAwesome
import argparse


import stitching_video

# Image frame sent to the Flask object
video_frame = None
pano = None

# Use locks for thread-safe viewing of frames in multiple browsers
thread_lock = threading.Lock()
pano_lock = threading.Lock()

# Argparse
modes = (cv2.Stitcher_PANORAMA, cv2.Stitcher_SCANS)


# GStreamer Pipeline to access the Raspberry Pi camera
#GSTREAMER_PIPELINE = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink wait-on-eos=false max-buffers=1 drop=True'

# Create the Flask object for the application
app = Flask(__name__)
fa = FontAwesome(app)


def captureFrames():
    global video_frame, thread_lock
    

    # Video capturing from OpenCV
    #video_capture = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    video_capture = cv2.VideoCapture(0)
    print("Video capture initialized ...")
    while True and video_capture.isOpened():
        return_key, frame = video_capture.read()
        if not return_key:
            break
        # Create a copy of the frame and store it in the global variable,
        # with thread safe access
        with thread_lock:
            video_frame = frame.copy()
        

        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    
    print("Video capture released ...")
    video_capture.release()


        
def encodeFrame():
    global thread_lock
    while True:
        # Acquire thread_lock to access the global video_frame object
        with thread_lock: 
            global video_frame
            if video_frame is None:
                continue
            return_key, encoded_image = cv2.imencode(".jpg", video_frame)
            if not return_key:
                continue
        
        # Output image as a byte array
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encoded_image) + b'\r\n')

        # NB : return would just send the frame at the request time.


def encodeLeftFrame():
    while True:
        # Acquire thread_lock to access the global video_frame object
        if stitching_video.left_image is None:
            continue
        return_key, encoded_image = cv2.imencode(".jpg", stitching_video.left_image)
        if not return_key:
            continue
        
        # Output image as a byte array
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encoded_image) + b'\r\n')



def encodeRightFrame():
    while True:
        # Acquire thread_lock to access the global video_frame object
        if stitching_video.right_image is None:
            continue
        return_key, encoded_image = cv2.imencode(".jpg", stitching_video.right_image)
        if not return_key:
            continue
    
        # Output image as a byte array
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encoded_image) + b'\r\n')



def encodeStitchedFrame():
    global pano
    while True:
        # Acquire thread_lock to access the global video_frame object
        if stitching_video.pano is None:
            continue

        with stitching_video.final_camera.read_lock:
            pano = stitching_video.pano.copy()
        
        return_key, encoded_image = cv2.imencode(".jpg", pano)
        if not return_key:
            continue
        
        
        # Output image as a byte array
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + 
            encoded_image.tobytes() + b'\r\n\r\n')




@app.route("/", methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', button='play')
    elif request.method == 'POST':
        
        req = request.form
        if req.get('play') == '1':
            print("Play pressed.")
            if stitching_video.final_camera is not None:
                stitching_video.final_camera.save = True
                stitching_video.final_camera.to_estimate = True
                button = 'stop'
        elif req.get('stop') == '1':
            print("Stop pressed.")
            stitching_video.final_camera.save = False
            stitching_video.final_camera.out.release()
            button = 'play'
        return render_template('index.html', button=button)
    #return Response(encodeLeftFrame(), mimetype = "multipart/x-mixed-replace; boundary=frame")




@app.route("/left")
def left():
    return Response(encodeLeftFrame(), mimetype = "multipart/x-mixed-replace; boundary=frame")


@app.route("/right")
def right():
    return Response(encodeRightFrame(), mimetype = "multipart/x-mixed-replace; boundary=frame")


@app.route("/stitch")
def stitch():
    return Response(encodeStitchedFrame(), mimetype = "multipart/x-mixed-replace; boundary=frame")


""" Another Data Streaming example
In this example, the 1500 rows are generated when the request is received. Therefore the data is not kept in memory
"""
@app.route('/large.csv')
def generate_large_csv():
    def generate():
        for row in range(15000):
            yield ',' + 'e' + '\n'
    return Response(generate(), mimetype='text/csv')



def read_args():
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
    parser.add_argument('--nothread', action='store_true', help='Run the stitching without threads')
    parser.add_argument('--timer', action='store_true', help='Enable timer to evaluate performance')
    args = parser.parse_args()
    return parser,args


# check to see if this is the main thread of execution
# ex usage(local) : python3 stitch_streaming.py --interface none --videos ../inputs/left.mp4 ../inputs/right.mp4 --capture_width 640 --capture_height 480
# ex usage (jetson nx , small) : python3 stitch_streaming.py --capture_width 640 --capture_height 480 --interface usb --device0 0 --device1 1 --view
# ex usage (jetson nx, 4K) : python3 stitch_streaming.py --capture_width 3840 --capture_height 2160 --interface usb --device0 0 --device1 1 


if __name__ == '__main__':
    # Read args
    parser, args = read_args()

    # Create a thread and attach the method that captures the image frames, to it
    #process_thread = threading.Thread(target=captureFrames)
    #process_thread.daemon = True

    # Start the thread
    #process_thread.start()

    # Start stitching_video
    stitching_thread = threading.Thread(target=stitching_video.main, args=[args])
    stitching_thread.start()

    # start the Flask Web Application
    # While it can be run on any feasible IP, IP = 0.0.0.0 renders the web app on
    # the host machine's localhost and is discoverable by other machines on the same network 
    app.run("0.0.0.0", port="8000")
