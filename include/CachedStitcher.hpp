/*
 * CachedStitcher - GPU-Accelerated Real-Time Image Stitching
 *
 * This class extends OpenCV's Stitcher to cache transformation matrices
 * and maximize GPU utilization for real-time panoramic video stitching.
 *
 * Key optimizations:
 * - One-time estimateTransform with cached camera parameters
 * - Pre-computed GPU warp maps stored in device memory
 * - CUDA stream parallelization for concurrent processing
 * - Zero-copy memory for video stream inputs
 */

#ifndef CACHED_STITCHER_HPP
#define CACHED_STITCHER_HPP

#include <opencv2/stitching/stitcher.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>

#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
#include <opencv2/gpu/gpu.hpp>
#include <cuda_runtime.h>
#endif

#include <vector>
#include <memory>

namespace cv {

class CachedStitcher : public Stitcher
{
public:
    // Constructor with GPU preference
    CachedStitcher(bool try_use_gpu = true);
    ~CachedStitcher();

    // Static factory method for optimized defaults
    static CachedStitcher createOptimized(bool try_use_gpu = true);

    // Cache transformation matrices after one-time calibration
    Status cacheTransformations(InputArray images);
    Status cacheTransformations(InputArray images, const std::vector<std::vector<Rect> > &rois);

    // Fast panorama composition using cached transforms
    Status composePanoramaGPU(InputArray images, OutputArray pano);
    Status composePanoramaGPU(OutputArray pano);  // Use last provided images

    // Cache management
    void invalidateCache();
    bool isCached() const { return transforms_cached_; }
    size_t getCacheMemoryUsage() const;

    // Advanced GPU configuration
    void setNumCudaStreams(int num_streams) { num_cuda_streams_ = num_streams; }
    int getNumCudaStreams() const { return num_cuda_streams_; }
    void setUsePinnedMemory(bool use) { use_pinned_memory_ = use; }
    bool getUsePinnedMemory() const { return use_pinned_memory_; }

    // Performance monitoring
    struct PerformanceStats {
        double transform_cache_time_ms;
        double last_compose_time_ms;
        double gpu_memory_mb;
        int frames_processed;
        double avg_fps;
    };
    PerformanceStats getPerformanceStats() const { return perf_stats_; }
    void resetPerformanceStats();

protected:
    // Override base class methods for optimization
    virtual Status matchImages();
    virtual void estimateCameraParams();

private:
    // Cached transformation data
    struct TransformCache {
        std::vector<detail::CameraParams> cameras;
        std::vector<int> indices;
        std::vector<Size> full_img_sizes;
        double work_scale;
        double seam_scale;
        double warped_image_scale;
        double compose_work_aspect;

#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
        // GPU-specific cached data
        // Pre-uploaded source frames
        std::vector<gpu::GpuMat> gpu_images_src;
        std::vector<gpu::GpuMat> gpu_xmaps;
        std::vector<gpu::GpuMat> gpu_ymaps;
        std::vector<gpu::GpuMat> gpu_seam_masks;
        std::vector<gpu::GpuMat> gpu_weight_maps;
        std::vector<gpu::GpuMat> gpu_exposure_gains;  // Cached exposure compensation gains

        // Pre-allocated GPU buffers
        std::vector<gpu::GpuMat> gpu_images_warped;
        std::vector<gpu::GpuMat> gpu_masks_warped;
        gpu::GpuMat gpu_panorama;

        // CUDA streams for parallel processing
        std::vector<cudaStream_t> cuda_streams;

        // Texture binding state
        bool textures_bound;
#endif
    };

    // Cache initialization methods
    void initializeGPUCache(const std::vector<Mat>& images);
    void precomputeWarpMaps();
    void precomputeSeamMasks();
    void allocateGPUBuffers(const std::vector<Size>& sizes);

    // GPU composition methods
    void warpImagesGPU(const std::vector<Mat>& images);
    void applyExposureCompensationGPU(int img_idx);
    void blendImagesGPU();

    // Memory management
    void releaseCache();
    void ensureGPUMemory(size_t required_bytes);

    // Member variables
    std::unique_ptr<TransformCache> cache_;
    bool transforms_cached_;
    bool gpu_enabled_;
    int num_cuda_streams_;
    bool use_pinned_memory_;

    // Performance tracking
    PerformanceStats perf_stats_;
    int64 total_frames_processed_;
    double total_compose_time_ms_;

    // GPU warper instances
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    Ptr<detail::PlaneWarperGpu> plane_warper_gpu_;
    Ptr<detail::CylindricalWarperGpu> cylindrical_warper_gpu_;
    Ptr<detail::SphericalWarperGpu> spherical_warper_gpu_;
#endif
};

// Utility class for managing CUDA resources
class CudaResourceManager {
public:
    CudaResourceManager(int num_streams = 4);
    ~CudaResourceManager();

    cudaStream_t getStream(int idx) const;
    void synchronizeAll();
    void synchronizeStream(int idx);

    static size_t getAvailableGPUMemory();
    static bool checkGPUCapability(int major, int minor);

private:
    std::vector<cudaStream_t> streams_;
    void cleanup();
};

} // namespace cv

#endif // CACHED_STITCHER_HPP
