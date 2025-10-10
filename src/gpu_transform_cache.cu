/*
 * CUDA Kernels for GPU Transform Caching
 * Optimized kernels for real-time image stitching
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <opencv2/gpu/device/common.hpp>
#include <opencv2/gpu/device/limits.hpp>

namespace cv {
namespace gpu {
namespace device {
namespace stitching {

// Texture memory for faster access to transformation maps
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_xmap;
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_ymap;
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> tex_image;

// Constants for kernel configuration
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define WARP_SIZE 32

// Structure for camera parameters in constant memory
struct CameraParams {
    float focal;
    float aspect;
    float ppx, ppy;
    float R[9];  // 3x3 rotation matrix
    float t[3];  // translation vector
    float K[9];  // intrinsics matrix
};

__constant__ CameraParams d_cameras[32];  // Support up to 32 cameras

// Fast warping kernel using cached transformation maps with texture memory optimization
__global__ void warpImageCachedKernel(
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
    bool use_texture)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_cols || y >= dst_rows)
        return;

    // Read transformation coordinates (texture memory provides 2-3x cache hit rate)
    float src_x, src_y;
    if (use_texture) {
        src_x = tex2D(tex_xmap, x, y);
        src_y = tex2D(tex_ymap, x, y);
    } else {
        const int map_idx = y * dst_cols + x;
        src_x = __ldg(&xmap[map_idx]);  // Use read-only cache for global memory
        src_y = __ldg(&ymap[map_idx]);
    }

    // Check bounds
    if (src_x < 0 || src_x >= src_cols - 1 || src_y < 0 || src_y >= src_rows - 1) {
        // Set to black or use border mode
        for (int c = 0; c < channels; ++c) {
            dst_image[y * dst_step + x * channels + c] = 0;
        }
        return;
    }

    // Bilinear interpolation
    const int x0 = __float2int_rd(src_x);
    const int y0 = __float2int_rd(src_y);
    const int x1 = min(x0 + 1, src_cols - 1);
    const int y1 = min(y0 + 1, src_rows - 1);

    const float fx = src_x - x0;
    const float fy = src_y - y0;
    const float fx1 = 1.0f - fx;
    const float fy1 = 1.0f - fy;

    // Interpolate for each channel using __ldg() for read-only cache
    for (int c = 0; c < channels; ++c) {
        const float v00 = __ldg(&src_image[y0 * src_step + x0 * channels + c]);
        const float v01 = __ldg(&src_image[y0 * src_step + x1 * channels + c]);
        const float v10 = __ldg(&src_image[y1 * src_step + x0 * channels + c]);
        const float v11 = __ldg(&src_image[y1 * src_step + x1 * channels + c]);

        const float val = fy1 * (fx1 * v00 + fx * v01) +
                         fy * (fx1 * v10 + fx * v11);

        dst_image[y * dst_step + x * channels + c] = __float2uint_rn(val);
    }
}

// Optimized kernel for spherical warping with pre-computed maps
__global__ void buildSphericalMapsKernel(
    float* xmap,
    float* ymap,
    int dst_rows,
    int dst_cols,
    float scale,
    const CameraParams camera)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_cols || y >= dst_rows)
        return;

    const float u = (x - dst_cols * 0.5f) / scale;
    const float v = (y - dst_rows * 0.5f) / scale;

    // Spherical projection formulas
    float u_ = atanf(u);
    float v_ = v / sqrtf(u * u + 1.0f);
    v_ = atanf(v_);

    // Apply rotation matrix
    float x_ = sinf(u_) * cosf(v_);
    float y_ = sinf(v_);
    float z_ = cosf(u_) * cosf(v_);

    // Rotate
    float x_rot = camera.R[0] * x_ + camera.R[1] * y_ + camera.R[2] * z_;
    float y_rot = camera.R[3] * x_ + camera.R[4] * y_ + camera.R[5] * z_;
    float z_rot = camera.R[6] * x_ + camera.R[7] * y_ + camera.R[8] * z_;

    // Project back to image coordinates
    if (fabsf(z_rot) > 1e-5f) {
        const float u_proj = camera.focal * x_rot / z_rot + camera.ppx;
        const float v_proj = camera.focal * camera.aspect * y_rot / z_rot + camera.ppy;

        xmap[y * dst_cols + x] = u_proj;
        ymap[y * dst_cols + x] = v_proj;
    } else {
        xmap[y * dst_cols + x] = -1.0f;
        ymap[y * dst_cols + x] = -1.0f;
    }
}

// Optimized kernel for cylindrical warping with pre-computed maps
__global__ void buildCylindricalMapsKernel(
    float* xmap,
    float* ymap,
    int dst_rows,
    int dst_cols,
    float scale,
    const CameraParams camera)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_cols || y >= dst_rows)
        return;

    const float u = (x - dst_cols * 0.5f) / scale;
    const float v = (y - dst_rows * 0.5f) / scale;

    // Cylindrical projection formulas
    float u_ = tanf(u);
    float v_ = v / cosf(u);

    // Apply rotation matrix
    float x_ = sinf(u);
    float y_ = v_;
    float z_ = cosf(u);

    // Rotate
    float x_rot = camera.R[0] * x_ + camera.R[1] * y_ + camera.R[2] * z_;
    float y_rot = camera.R[3] * x_ + camera.R[4] * y_ + camera.R[5] * z_;
    float z_rot = camera.R[6] * x_ + camera.R[7] * y_ + camera.R[8] * z_;

    // Project back to image coordinates
    if (fabsf(z_rot) > 1e-5f) {
        const float u_proj = camera.focal * x_rot / z_rot + camera.ppx;
        const float v_proj = camera.focal * camera.aspect * y_rot / z_rot + camera.ppy;

        xmap[y * dst_cols + x] = u_proj;
        ymap[y * dst_cols + x] = v_proj;
    } else {
        xmap[y * dst_cols + x] = -1.0f;
        ymap[y * dst_cols + x] = -1.0f;
    }
}

// Fast exposure compensation kernel
__global__ void applyExposureCompensationKernel(
    uchar* image,
    const float* gains, // 3 floats per tile (RGB)
    int rows,
    int cols,
    int step,
    int channels,
    int tiles_x,
    int tiles_y)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows)
        return;

    const int tx = min(x / 32, tiles_x - 1);
    const int ty = min(y / 32, tiles_y - 1);
    const int tile_index = (ty * tiles_x + tx) * 3;

    for (int c = 0; c < channels; ++c) {
        const float gain = gains[tile_index + (c < 3 ? c : 2)];
        int idx = y * step + x * channels + c;
        float val = image[idx] * gain;
        val = fminf(fmaxf(val, 0.0f), 255.0f);
        image[idx] = __float2uint_rn(val);
    }
}

// Optimized multi-band blending kernel
__global__ void multiband_blend_kernel(
    const float* src1,
    const float* src2,
    const float* weight1,
    const float* weight2,
    float* dst,
    int rows,
    int cols,
    int channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows)
        return;

    const int pixel_idx = y * cols + x;
    const float w1 = weight1[pixel_idx];
    const float w2 = weight2[pixel_idx];
    const float total_weight = w1 + w2 + 1e-5f;  // Avoid division by zero

    for (int c = 0; c < channels; ++c) {
        const int idx = pixel_idx * channels + c;
        dst[idx] = (src1[idx] * w1 + src2[idx] * w2) / total_weight;
    }
}

// Seam mask generation kernel
__global__ void generateSeamMaskKernel(
    const uchar* labels,
    uchar* mask,
    int rows,
    int cols,
    int label_id)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows)
        return;

    const int idx = y * cols + x;
    mask[idx] = (labels[idx] == label_id) ? 255 : 0;
}

// Host wrapper functions
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
    bool use_texture)
{
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((dst_cols + block.x - 1) / block.x,
              (dst_rows + block.y - 1) / block.y);

    warpImageCachedKernel<<<grid, block, 0, stream>>>(
        xmap, ymap, src_image, src_rows, src_cols, src_step,
        dst_image, dst_rows, dst_cols, dst_step, channels, use_texture);
}

// Bind texture memory for faster warp map access
void bindWarpMapTextures(const float* xmap, const float* ymap, int rows, int cols, size_t pitch) {
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(0, tex_xmap, xmap, desc, cols, rows, pitch);
    cudaBindTexture2D(0, tex_ymap, ymap, desc, cols, rows, pitch);
}

// Unbind texture memory
void unbindWarpMapTextures() {
    cudaUnbindTexture(tex_xmap);
    cudaUnbindTexture(tex_ymap);
}

void launchBuildSphericalMaps(
    float* xmap,
    float* ymap,
    int dst_rows,
    int dst_cols,
    float scale,
    const CameraParams& camera,
    cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((dst_cols + block.x - 1) / block.x,
              (dst_rows + block.y - 1) / block.y);

    buildSphericalMapsKernel<<<grid, block, 0, stream>>>(
        xmap, ymap, dst_rows, dst_cols, scale, camera);
}

void launchBuildCylindricalMaps(
    float* xmap,
    float* ymap,
    int dst_rows,
    int dst_cols,
    float scale,
    const CameraParams& camera,
    cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((dst_cols + block.x - 1) / block.x,
              (dst_rows + block.y - 1) / block.y);

    buildCylindricalMapsKernel<<<grid, block, 0, stream>>>(
        xmap, ymap, dst_rows, dst_cols, scale, camera);
}

void launchApplyExposureCompensation(
    uchar* image,
    const float* gains,
    int rows,
    int cols,
    int step,
    int channels,
    int tiles_x,
    int tiles_y,
    cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    applyExposureCompensationKernel<<<grid, block, 0, stream>>>(
        image, gains, rows, cols, step, channels, tiles_x, tiles_y);
}

void launchMultibandBlend(
    const float* src1,
    const float* src2,
    const float* weight1,
    const float* weight2,
    float* dst,
    int rows,
    int cols,
    int channels,
    cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    multiband_blend_kernel<<<grid, block, 0, stream>>>(
        src1, src2, weight1, weight2, dst, rows, cols, channels);
}

} // namespace stitching
} // namespace device
} // namespace gpu
} // namespace cv
