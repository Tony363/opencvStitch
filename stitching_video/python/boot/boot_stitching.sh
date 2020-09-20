#!/bin/bash

# Log current video cameras detected
ls -l /dev/video* | tee cam.txt

# Initialize working folder and venv
cd ../
WORK_FOLDER=$(pwd)

#VENV=$WORK_FOLDER"/stitch-venv/bin/activate"

STREAM_STITCHING=$WORK_FOLDER"/stitch_streaming.py"


# Activate venv (must be disabled on JetsonNX if using local environment)
#source $VENV

# Find cameras
#echo ll /dev/ | rev | cut -d' ' -f1 | rev | grep video*
c=0
for video in $(ls -la /dev/ | rev | cut -d' ' -f1 | rev | grep video*); do
	eval "video$c=${video: -1}";
	echo ${video: -1}
	c=$((c+1));
done
 
# Run stream stitching with Flask server
if [ -z $1 ] || [ $1 == "high" ]
then
echo 'Resolution set to HIGH : 3840x2160'
python3 $STREAM_STITCHING --capture_width 3840 --capture_height 2160 --interface usb --device0 $video0 --device1 $video1
elif [ $1 == "low" ]
then
echo 'Resolution set to LOW : 640x480'
python3 $STREAM_STITCHING --capture_width 640 --capture_height 480 --interface usb --device0 $video0 --device1 $video1
fi