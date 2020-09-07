#!/bin/bash

# Log current video cameras detected
ls -l /dev/video* | tee cam.txt

# Initialize working folder and venv
cd ../
WORK_FOLDER=$(pwd)

VENV=$WORK_FOLDER"/stitch-venv/bin/activate"

STREAM_STITCHING=$WORK_FOLDER"/stitch_streaming.py"


# Activate venv (must be disabled on JetsonNX if using local environment)
#source $VENV

# Run stream stitching with Flask server

if [ -z $1 ] || [ $1 == "high" ]
then
echo 'Resolution set to HIGH : 3840x2160'
python3 $STREAM_STITCHING --capture_width 3840 --capture_height 2160 --interface usb --device0 0 --device1 1
elif [ $1 == "low" ]
then
echo 'Resolution set to LOW : 640x480'
python3 $STREAM_STITCHING --capture_width 640 --capture_height 480 --interface usb --device0 0 --device1 1
fi
