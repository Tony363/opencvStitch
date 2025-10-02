# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is OpenCV 2.4.x - an open source computer vision library with image stitching capabilities. The codebase is a large C++ library organized into modules.

## Build Commands

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the library
make -j$(nproc)

# Run tests
make test

# Build specific module
make opencv_stitching

# Build and run a single test
./bin/opencv_test_stitching --gtest_filter=TestName
```

## Core Architecture

### Module Organization
The library is divided into modules under `modules/`:
- **core**: Basic structures, algorithms, and utility functions
- **imgproc**: Image processing functions
- **highgui**: UI and image/video I/O
- **features2d**: Feature detection and description
- **calib3d**: Camera calibration and 3D reconstruction
- **stitching**: Image stitching pipeline
- **gpu**: GPU-accelerated implementations
- **ocl**: OpenCL implementations

### Stitching Module Architecture
The stitching module (`modules/stitching/`) implements a complete panorama creation pipeline:
- **Stitcher class** (`include/opencv2/stitching/stitcher.hpp`): High-level API for image stitching
- **Detail namespace** (`include/opencv2/stitching/detail/`): Low-level components:
  - `matchers.hpp`: Feature matching algorithms
  - `motion_estimators.hpp`: Camera motion estimation
  - `warpers.hpp`: Image warping for different projections
  - `exposure_compensate.hpp`: Exposure compensation
  - `seam_finders.hpp`: Seam finding for blending
  - `blenders.hpp`: Image blending algorithms

### Key Design Patterns
- **Builder pattern**: Stitcher class uses builder pattern for configuration
- **Pipeline architecture**: Stitching process is a configurable pipeline of stages
- **GPU acceleration**: Many algorithms have both CPU and GPU implementations
- **Factory methods**: Component creation through factory methods for flexibility

## Sample Applications

- `samples/cpp/stitching.cpp`: Simple stitching demo using high-level API
- `samples/cpp/stitching_detailed.cpp`: Advanced demo with pipeline customization

## Testing

Tests are located in `modules/*/test/` directories. The stitching module tests are in `modules/stitching/test/`.

## Important Conventions

- OpenCV 2.4 uses `cv` namespace
- Mat is the primary image container
- Functions often have multiple overloads for different data types
- GPU functions are in separate gpu module with similar API