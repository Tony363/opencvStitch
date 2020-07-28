# Install OpenCV with modified modules from sources for Python usage
# reference: https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/

# apt-get install build-essential git python3 python3-dev cmake

# 1. Create python virtual environment
cd stitching_video/python
python3 -m venv stitch-venv
cd stitch-venv/bin
source activate
pip install numpy

PYTHON_PATH=$(pwd)"/python"

# 2. Install OpenCV

cd ../../../../
rm -rf build
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D PYTHON_EXECUTABLE=$PYTHON_PATH ..

#cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_ENABLE_NONFREE=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D BUILD_opencv_python3=ON -D HAVE_opencv_python3=ON -D PYTHON_EXECUTABLE=/opencv-stitch/stitching_video/python/stitch-venv/bin/python3 ..




make -j4
sudo make install
sudo ldconfig


# 3. Link OpenCV into python (Python 3.6)
sudo mv /usr/local/lib/python3.6/dist-packages/cv2/python-3.6/cv2.cpython-36m-x86_64-linux-gnu.so cv2.so
cd stitching_video/python/stitch-venv/lib/python3.6/site-packages/
ln -s /usr/local/lib/python3.6/dist-packages/cv2/python-3.6/cv2.so cv2.so


