#!/bin/bash

# Show cameras (video0 and video1) available capture width and height
# Display in stdout and Write resolutions in text file (cam1.txt and cam2.txt).
# And the available FPS

DEVICE_1=/dev/video$1
DEVICE_2=/dev/video$2

echo get list formats of $DEVICE_1
v4l2-ctl -d $DEVICE_1 --list-formats-ext | tee cam0.txt

echo get list formats of $DEVICE_2
v4l2-ctl -d $DEVICE_2 --list-formats-ext | tee cam1.txt

# Use MJPG decoder to increase fps
# https://answers.opencv.org/question/41899/changing-pixel-format-yuyv-to-mjpg-when-capturing-from-webcam/