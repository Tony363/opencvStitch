/*
 * Benchmark for CachedStitcher
 * Usage:
 *   ./benchmark_stitching --gpu [--iters 200] img1.jpg img2.jpg [...]
 */

#include "../include/CachedStitcher.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>

using namespace cv;
using namespace std;

struct Args {
    bool use_gpu = true;
    int iters = 200;
    vector<string> images;
};

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        string s = argv[i];
        if (s == "--gpu") a.use_gpu = true;
        else if (s == "--no-gpu") a.use_gpu = false;
        else if (s == "--iters" && i+1 < argc) a.iters = atoi(argv[++i]);
        else if (!s.empty() && s[0] != '-') a.images.push_back(s);
    }
    return a;
}

int main(int argc, char** argv) {
    Args args = parse(argc, argv);
    if (args.images.size() < 2) {
        cerr << "Provide at least two images." << endl;
        return 1;
    }

    // Load images
    vector<Mat> imgs;
    for (auto& p : args.images) {
        Mat im = imread(p);
        if (im.empty()) { cerr << "Failed to load " << p << endl; return 1; }
        imgs.push_back(im);
    }

    // Create stitcher
    cv::CachedStitcher stitcher = cv::CachedStitcher::createOptimized(args.use_gpu);
    stitcher.setNumCudaStreams(4);
    stitcher.setUsePinnedMemory(true);

    // Calibrate/cache
    cout << "Calibrating..." << endl;
    auto t0 = chrono::high_resolution_clock::now();
    auto status = stitcher.cacheTransformations(imgs);
    auto t1 = chrono::high_resolution_clock::now();
    if (status != cv::CachedStitcher::OK) {
        cerr << "Calibration failed" << endl;
        return 1;
    }
    double calib_ms = chrono::duration<double, std::milli>(t1 - t0).count();
    cout << "Calibration: " << calib_ms << " ms\n";

    // Warmup
    Mat pano;
    stitcher.composePanoramaGPU(imgs, pano);

    // Benchmark loop
    vector<double> frame_ms;
    vector<double> upload_ms, warp_ms, exposure_ms, blend_ms, download_ms;
    frame_ms.reserve(args.iters);

    for (int i = 0; i < args.iters; ++i) {
        auto f0 = chrono::high_resolution_clock::now();
        stitcher.composePanoramaGPU(imgs, pano);
        auto f1 = chrono::high_resolution_clock::now();
        auto s = stitcher.getPerformanceStats();
        frame_ms.push_back(chrono::duration<double, std::milli>(f1 - f0).count());
        upload_ms.push_back(s.upload_time_ms);
        warp_ms.push_back(s.warp_time_ms);
        exposure_ms.push_back(s.exposure_time_ms);
        blend_ms.push_back(s.blend_time_ms);
        download_ms.push_back(s.download_time_ms);
    }

    auto avg = [](const vector<double>& v){
        return accumulate(v.begin(), v.end(), 0.0) / max<size_t>(1, v.size());
    };

    cout << fixed;
    cout << "Iters: " << args.iters << "\n";
    cout << "Avg frame: " << avg(frame_ms) << " ms (" << 1000.0/avg(frame_ms) << " FPS)\n";
    cout << "  Upload:   " << avg(upload_ms) << " ms\n";
    cout << "  Warp:     " << avg(warp_ms) << " ms\n";
    cout << "  Exposure: " << avg(exposure_ms) << " ms\n";
    cout << "  Blend:    " << avg(blend_ms) << " ms\n";
    cout << "  Download: " << avg(download_ms) << " ms\n";

    return 0;
}

