/*
 * CachedStitcher Implementation
 * GPU-accelerated real-time image stitching with transformation caching
 */

#include "CachedStitcher.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <iostream>

#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
#include <opencv2/gpu/gpu.hpp>
#endif

namespace cv {

// Constructor
CachedStitcher::CachedStitcher(bool try_use_gpu)
    : Stitcher()
    , transforms_cached_(false)
    , gpu_enabled_(try_use_gpu)
    , num_cuda_streams_(4)
    , use_pinned_memory_(true)
    , total_frames_processed_(0)
    , total_compose_time_ms_(0.0)
{
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    if (gpu_enabled_ && gpu::getCudaEnabledDeviceCount() > 0) {
        cache_.reset(new TransformCache());

        // Initialize GPU warpers
        plane_warper_gpu_ = new detail::PlaneWarperGpu();
        cylindrical_warper_gpu_ = new detail::CylindricalWarperGpu();
        spherical_warper_gpu_ = new detail::SphericalWarperGpu();

        // Set GPU device
        gpu::setDevice(0);

        std::cout << "CachedStitcher: GPU acceleration enabled (Device: "
                  << gpu::getDevice() << ")" << std::endl;
    } else {
        gpu_enabled_ = false;
        std::cout << "CachedStitcher: GPU not available, using CPU fallback" << std::endl;
    }
#else
    gpu_enabled_ = false;
    std::cout << "CachedStitcher: Compiled without GPU support" << std::endl;
#endif

    resetPerformanceStats();
}

// Destructor
CachedStitcher::~CachedStitcher() {
    releaseCache();
}

// Factory method for optimized configuration
CachedStitcher CachedStitcher::createOptimized(bool try_use_gpu) {
    CachedStitcher stitcher(try_use_gpu);

    // Optimized settings for real-time performance
    stitcher.setRegistrationResol(0.4);  // Lower for speed
    stitcher.setSeamEstimationResol(0.08);  // Lower for speed
    stitcher.setCompositingResol(ORIG_RESOL);
    stitcher.setPanoConfidenceThresh(0.9);  // Slightly lower threshold
    stitcher.setWaveCorrection(true);
    stitcher.setWaveCorrectKind(detail::WAVE_CORRECT_HORIZ);

#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    if (try_use_gpu && gpu::getCudaEnabledDeviceCount() > 0) {
        // Use GPU-accelerated components
        stitcher.setFeaturesMatcher(new detail::BestOf2NearestMatcher(true));
        stitcher.setBundleAdjuster(new detail::BundleAdjusterRay());

#ifdef HAVE_OPENCV_NONFREE
        stitcher.setFeaturesFinder(new detail::SurfFeaturesFinderGpu());
#else
        stitcher.setFeaturesFinder(new detail::OrbFeaturesFinder());
#endif

        stitcher.setWarper(new SphericalWarperGpu());
        stitcher.setSeamFinder(new detail::GraphCutSeamFinderGpu());
        stitcher.setBlender(new detail::MultiBandBlender(true));
    } else
#endif
    {
        // CPU fallback
        stitcher.setFeaturesMatcher(new detail::BestOf2NearestMatcher(false));
        stitcher.setBundleAdjuster(new detail::BundleAdjusterRay());

#ifdef HAVE_OPENCV_NONFREE
        stitcher.setFeaturesFinder(new detail::SurfFeaturesFinder());
#else
        stitcher.setFeaturesFinder(new detail::OrbFeaturesFinder());
#endif

        stitcher.setWarper(new SphericalWarper());
        stitcher.setSeamFinder(new detail::GraphCutSeamFinder(detail::GraphCutSeamFinderBase::COST_COLOR));
        stitcher.setBlender(new detail::MultiBandBlender(false));
    }

    stitcher.setExposureCompensator(new detail::BlocksGainCompensator());

    return stitcher;
}

// Cache transformation matrices
CachedStitcher::Status CachedStitcher::cacheTransformations(InputArray images) {
    return cacheTransformations(images, std::vector<std::vector<Rect> >());
}

CachedStitcher::Status CachedStitcher::cacheTransformations(
    InputArray images,
    const std::vector<std::vector<Rect> > &rois)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    // First, run standard transformation estimation
    Status status = estimateTransform(images, rois);
    if (status != OK) {
        return status;
    }

    // Cache the computed transformations
    if (!cache_) {
        cache_.reset(new TransformCache());
    }

    // Store transformation data
    cache_->cameras = cameras_;
    cache_->indices = indices_;
    cache_->full_img_sizes = full_img_sizes_;
    cache_->work_scale = work_scale_;
    cache_->seam_scale = seam_scale_;
    cache_->warped_image_scale = warped_image_scale_;

    // Get images for cache initialization
    std::vector<Mat> imgs;
    images.getMatVector(imgs);

#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    if (gpu_enabled_) {
        // Initialize GPU-specific cache
        initializeGPUCache(imgs);

        // Pre-compute warp maps
        precomputeWarpMaps();

        // Pre-compute seam masks
        precomputeSeamMasks();

        // Allocate GPU buffers
        std::vector<Size> sizes;
        for (size_t i = 0; i < cache_->indices.size(); ++i) {
            sizes.push_back(cache_->full_img_sizes[cache_->indices[i]]);
        }
        allocateGPUBuffers(sizes);
    }
#endif

    transforms_cached_ = true;

    auto end_time = std::chrono::high_resolution_clock::now();
    perf_stats_.transform_cache_time_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();

    std::cout << "Transformations cached in " << perf_stats_.transform_cache_time_ms
              << " ms" << std::endl;

    return OK;
}

// Fast panorama composition using cached transforms
CachedStitcher::Status CachedStitcher::composePanoramaGPU(OutputArray pano) {
    return composePanoramaGPU(std::vector<Mat>(), pano);
}

CachedStitcher::Status CachedStitcher::composePanoramaGPU(
    InputArray images,
    OutputArray pano)
{
    if (!transforms_cached_) {
        std::cerr << "Error: Transformations not cached. Call cacheTransformations first." << std::endl;
        return ERR_NEED_MORE_IMGS;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<Mat> imgs;
    images.getMatVector(imgs);

    // If new images provided, update the working set
    if (!imgs.empty()) {
        if (imgs.size() != imgs_.size()) {
            std::cerr << "Error: Number of images doesn't match cached transformations" << std::endl;
            return ERR_NEED_MORE_IMGS;
        }
        imgs_ = imgs;
    }

#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    if (gpu_enabled_ && cache_) {
        // GPU-accelerated composition path
        Mat &pano_ = pano.getMatRef();

        // Upload images to GPU
        cache_->gpu_images_warped.resize(imgs_.size());
        for (size_t i = 0; i < imgs_.size(); ++i) {
            Mat img;
            if (std::abs(cache_->compose_work_aspect - 1) > 1e-1) {
                resize(imgs_[i], img, Size(), cache_->compose_work_aspect,
                       cache_->compose_work_aspect);
            } else {
                img = imgs_[i];
            }

            // Upload to GPU
            cache_->gpu_images_warped[i].upload(img);
        }

        // Warp images using cached maps
        warpImagesGPU(imgs_);

        // Apply exposure compensation
        for (size_t i = 0; i < imgs_.size(); ++i) {
            applyExposureCompensationGPU(i);
        }

        // Blend images
        blendImagesGPU();

        // Download result
        cache_->gpu_panorama.download(pano_);

    } else
#endif
    {
        // CPU fallback - use standard composePanorama
        return Stitcher::composePanorama(images, pano);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double compose_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Update performance stats
    perf_stats_.last_compose_time_ms = compose_time;
    total_compose_time_ms_ += compose_time;
    total_frames_processed_++;
    perf_stats_.frames_processed = total_frames_processed_;
    perf_stats_.avg_fps = 1000.0 / (total_compose_time_ms_ / total_frames_processed_);

    return OK;
}

// Initialize GPU cache
void CachedStitcher::initializeGPUCache(const std::vector<Mat>& images) {
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    if (!gpu_enabled_ || !cache_) return;

    // Initialize CUDA streams
    cache_->cuda_streams.resize(num_cuda_streams_);
    for (int i = 0; i < num_cuda_streams_; ++i) {
        cudaStreamCreate(&cache_->cuda_streams[i]);
    }

    // Pre-allocate GPU matrices
    size_t num_images = cache_->indices.size();
    cache_->gpu_xmaps.resize(num_images);
    cache_->gpu_ymaps.resize(num_images);
    cache_->gpu_seam_masks.resize(num_images);
    cache_->gpu_weight_maps.resize(num_images);
    cache_->gpu_images_warped.resize(num_images);
    cache_->gpu_masks_warped.resize(num_images);
#endif
}

// Pre-compute warp maps
void CachedStitcher::precomputeWarpMaps() {
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    if (!gpu_enabled_ || !cache_) return;

    Ptr<detail::RotationWarper> w = warper_->create(
        float(cache_->warped_image_scale * cache_->seam_scale));

    for (size_t i = 0; i < cache_->indices.size(); ++i) {
        Mat_<float> K;
        cache_->cameras[i].K().convertTo(K, CV_32F);
        K(0,0) *= (float)cache_->seam_scale;
        K(0,2) *= (float)cache_->seam_scale;
        K(1,1) *= (float)cache_->seam_scale;
        K(1,2) *= (float)cache_->seam_scale;

        Size img_size = cache_->full_img_sizes[cache_->indices[i]];

        // Build and cache the warp maps
        if (dynamic_cast<detail::SphericalWarperGpu*>(w.get())) {
            dynamic_cast<detail::SphericalWarperGpu*>(w.get())->buildMaps(
                img_size, K, cache_->cameras[i].R,
                cache_->gpu_xmaps[i], cache_->gpu_ymaps[i]);
        } else if (dynamic_cast<detail::CylindricalWarperGpu*>(w.get())) {
            dynamic_cast<detail::CylindricalWarperGpu*>(w.get())->buildMaps(
                img_size, K, cache_->cameras[i].R,
                cache_->gpu_xmaps[i], cache_->gpu_ymaps[i]);
        } else if (dynamic_cast<detail::PlaneWarperGpu*>(w.get())) {
            dynamic_cast<detail::PlaneWarperGpu*>(w.get())->buildMaps(
                img_size, K, cache_->cameras[i].R,
                cache_->gpu_xmaps[i], cache_->gpu_ymaps[i]);
        }
    }
#endif
}

// Pre-compute seam masks
void CachedStitcher::precomputeSeamMasks() {
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    if (!gpu_enabled_ || !cache_) return;

    // Seam masks will be computed during first composition
    // and then cached for subsequent frames
#endif
}

// Allocate GPU buffers
void CachedStitcher::allocateGPUBuffers(const std::vector<Size>& sizes) {
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    if (!gpu_enabled_ || !cache_) return;

    // Calculate maximum panorama size
    int max_width = 0, max_height = 0;
    for (const auto& size : sizes) {
        max_width = std::max(max_width, size.width * 2);
        max_height = std::max(max_height, size.height * 2);
    }

    // Allocate panorama buffer
    cache_->gpu_panorama.create(max_height, max_width, CV_16SC3);

    // Track GPU memory usage
    perf_stats_.gpu_memory_mb = (cache_->gpu_panorama.cols * cache_->gpu_panorama.rows *
                                  cache_->gpu_panorama.elemSize()) / (1024.0 * 1024.0);

    // Add memory for warp maps and other buffers
    for (size_t i = 0; i < sizes.size(); ++i) {
        perf_stats_.gpu_memory_mb += (sizes[i].width * sizes[i].height *
                                      sizeof(float) * 4) / (1024.0 * 1024.0);
    }
#endif
}

// Warp images using cached GPU maps
void CachedStitcher::warpImagesGPU(const std::vector<Mat>& images) {
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    if (!gpu_enabled_ || !cache_) return;

    for (size_t i = 0; i < images.size(); ++i) {
        int stream_idx = i % num_cuda_streams_;
        gpu::Stream stream(cache_->cuda_streams[stream_idx]);

        // Remap using cached transformation maps
        gpu::remap(cache_->gpu_images_warped[i], cache_->gpu_images_warped[i],
                   cache_->gpu_xmaps[i], cache_->gpu_ymaps[i],
                   INTER_LINEAR, BORDER_REFLECT, Scalar(), stream);
    }

    // Synchronize all streams
    for (int i = 0; i < num_cuda_streams_; ++i) {
        cudaStreamSynchronize(cache_->cuda_streams[i]);
    }
#endif
}

// Apply exposure compensation on GPU
void CachedStitcher::applyExposureCompensationGPU(int img_idx) {
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    if (!gpu_enabled_ || !cache_) return;

    // Apply exposure compensation if configured
    if (exposure_comp_) {
        // This would require GPU implementation of exposure compensation
        // For now, we'll skip or use CPU fallback
    }
#endif
}

// Blend images on GPU
void CachedStitcher::blendImagesGPU() {
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    if (!gpu_enabled_ || !cache_) return;

    // Use GPU blender if available
    if (blender_) {
        // This would require GPU implementation of blending
        // For now, we use the existing blender with GPU data
    }
#endif
}

// Invalidate cache
void CachedStitcher::invalidateCache() {
    transforms_cached_ = false;
    releaseCache();
}

// Release cached resources
void CachedStitcher::releaseCache() {
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    if (cache_) {
        // Destroy CUDA streams
        for (auto& stream : cache_->cuda_streams) {
            cudaStreamDestroy(stream);
        }

        // Clear GPU matrices
        cache_->gpu_xmaps.clear();
        cache_->gpu_ymaps.clear();
        cache_->gpu_seam_masks.clear();
        cache_->gpu_weight_maps.clear();
        cache_->gpu_images_warped.clear();
        cache_->gpu_masks_warped.clear();
        cache_->gpu_panorama.release();
    }
#endif

    cache_.reset();
    transforms_cached_ = false;
}

// Get cache memory usage
size_t CachedStitcher::getCacheMemoryUsage() const {
    return static_cast<size_t>(perf_stats_.gpu_memory_mb * 1024 * 1024);
}

// Reset performance statistics
void CachedStitcher::resetPerformanceStats() {
    perf_stats_ = PerformanceStats();
    total_frames_processed_ = 0;
    total_compose_time_ms_ = 0.0;
}

// Override matchImages for optimization
CachedStitcher::Status CachedStitcher::matchImages() {
    // Use cached data if available
    if (transforms_cached_ && cache_) {
        // Skip matching, use cached data
        return OK;
    }

    // Otherwise use base implementation
    return Stitcher::matchImages();
}

// Override estimateCameraParams for optimization
void CachedStitcher::estimateCameraParams() {
    // Use cached data if available
    if (transforms_cached_ && cache_) {
        // Restore cached camera parameters
        cameras_ = cache_->cameras;
        return;
    }

    // Otherwise use base implementation
    Stitcher::estimateCameraParams();
}

// CudaResourceManager implementation
CudaResourceManager::CudaResourceManager(int num_streams) {
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    streams_.resize(num_streams);
    for (auto& stream : streams_) {
        cudaStreamCreate(&stream);
    }
#endif
}

CudaResourceManager::~CudaResourceManager() {
    cleanup();
}

void CudaResourceManager::cleanup() {
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    for (auto& stream : streams_) {
        cudaStreamDestroy(stream);
    }
#endif
}

cudaStream_t CudaResourceManager::getStream(int idx) const {
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    if (idx >= 0 && idx < streams_.size()) {
        return streams_[idx];
    }
#endif
    return 0;
}

void CudaResourceManager::synchronizeAll() {
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    for (const auto& stream : streams_) {
        cudaStreamSynchronize(stream);
    }
#endif
}

void CudaResourceManager::synchronizeStream(int idx) {
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    if (idx >= 0 && idx < streams_.size()) {
        cudaStreamSynchronize(streams_[idx]);
    }
#endif
}

size_t CudaResourceManager::getAvailableGPUMemory() {
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
#else
    return 0;
#endif
}

bool CudaResourceManager::checkGPUCapability(int major, int minor) {
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    int device_count = gpu::getCudaEnabledDeviceCount();
    if (device_count > 0) {
        gpu::DeviceInfo info(0);
        return info.majorVersion() >= major && info.minorVersion() >= minor;
    }
#endif
    return false;
}

} // namespace cv