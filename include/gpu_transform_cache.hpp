/*
 * GPU Transform Cache CUDA Kernel Declarations
 * Header file for GPU-accelerated image stitching kernels
 */

#ifndef GPU_TRANSFORM_CACHE_HPP
#define GPU_TRANSFORM_CACHE_HPP

#include <cuda_runtime.h>

namespace cv {
namespace gpu {
namespace device {
namespace stitching {

// Structure for camera parameters
struct CameraParams {
    float focal;
    float aspect;
    float ppx, ppy;
    float R[9];
    float t[3];
    float K[9];
};

// Warp map texture binding
void bindWarpMapTextures(const float* xmap, const float* ymap, int rows, int cols, size_t pitch);
void unbindWarpMapTextures();

// Core warping kernel
void launchWarpImageCached(
    const float* xmap,
    const float* ymap,
    const uchar* src_image,
    int src_rows,
    int src_cols,
    int src_step,
    uchar* dst_image,
    int dst_rows,
    int dst_cols,
    int dst_step,
    int channels,
    cudaStream_t stream,
    bool use_texture = false);

// Map building kernels
void launchBuildSphericalMaps(
    float* xmap,
    float* ymap,
    int dst_rows,
    int dst_cols,
    float scale,
    const CameraParams& camera,
    cudaStream_t stream);

void launchBuildCylindricalMaps(
    float* xmap,
    float* ymap,
    int dst_rows,
    int dst_cols,
    float scale,
    const CameraParams& camera,
    cudaStream_t stream);

// Exposure compensation
void launchApplyExposureCompensation(
    uchar* image,
    const float* gains, // 3-channel per-tile grid (CV_32FC3 layout)
    int rows,
    int cols,
    int step,
    int channels,
    int tiles_x,
    int tiles_y,
    cudaStream_t stream);

// Multi-band blending
void launchMultibandBlend(
    const float* src1,
    const float* src2,
    const float* weight1,
    const float* weight2,
    float* dst,
    int rows,
    int cols,
    int channels,
    cudaStream_t stream);

} // namespace stitching
} // namespace device
} // namespace gpu
} // namespace cv

#endif // GPU_TRANSFORM_CACHE_HPP
