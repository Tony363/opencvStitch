/*
 * Real-time Image Stitching Example
 * Demonstrates GPU-accelerated panoramic video stitching using CachedStitcher
 *
 * Usage:
 * ./realtime_stitching [options] img1 img2 img3 ...
 *   or
 * ./realtime_stitching --video cam1.mp4 cam2.mp4 cam3.mp4 ...
 *   or
 * ./realtime_stitching --camera 0 1 2 ...  (for webcams)
 *
 * Options:
 *   --gpu          Enable GPU acceleration (default: true)
 *   --output       Output video file (default: display only)
 *   --fps          Target FPS (default: 30)
 *   --resolution   Output resolution (720p, 1080p, 4k)
 *   --calibrate    Number of frames for calibration (default: 10)
 */

#include "../include/CachedStitcher.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace cv;
using namespace std;

// Command-line argument parser
struct Config {
    bool use_gpu = true;
    bool use_video = false;
    bool use_camera = false;
    vector<string> input_sources;
    string output_file;
    int target_fps = 30;
    string resolution = "1080p";
    int calibration_frames = 10;
    bool show_fps = true;
    bool debug_mode = false;
    bool bench_mode = false;
};

Config parseArgs(int argc, char** argv) {
    Config config;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];

        if (arg == "--gpu") {
            config.use_gpu = true;
        } else if (arg == "--no-gpu") {
            config.use_gpu = false;
        } else if (arg == "--video") {
            config.use_video = true;
        } else if (arg == "--camera") {
            config.use_camera = true;
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_file = argv[++i];
        } else if (arg == "--fps" && i + 1 < argc) {
            config.target_fps = atoi(argv[++i]);
        } else if (arg == "--resolution" && i + 1 < argc) {
            config.resolution = argv[++i];
        } else if (arg == "--calibrate" && i + 1 < argc) {
            config.calibration_frames = atoi(argv[++i]);
        } else if (arg == "--debug") {
            config.debug_mode = true;
        } else if (arg == "--bench") {
            config.bench_mode = true;
        } else if (arg[0] != '-') {
            config.input_sources.push_back(arg);
        }
    }

    return config;
}

// FPS counter class
class FPSCounter {
private:
    int frame_count = 0;
    double total_time = 0;
    chrono::high_resolution_clock::time_point last_time;
    double current_fps = 0;

public:
    void start() {
        last_time = chrono::high_resolution_clock::now();
    }

    void tick() {
        auto now = chrono::high_resolution_clock::now();
        double delta = chrono::duration<double>(now - last_time).count();
        total_time += delta;
        frame_count++;

        if (total_time >= 1.0) {  // Update FPS every second
            current_fps = frame_count / total_time;
            frame_count = 0;
            total_time = 0;
        }

        last_time = now;
    }

    double getFPS() const { return current_fps; }

    void display(Mat& image) {
        stringstream ss;
        ss << fixed << setprecision(1) << "FPS: " << current_fps;
        putText(image, ss.str(), Point(10, 30), FONT_HERSHEY_SIMPLEX,
                1.0, Scalar(0, 255, 0), 2);
    }
};

// Video source wrapper
class VideoSource {
private:
    vector<VideoCapture> captures;
    vector<Mat> current_frames;

public:
    bool open(const vector<string>& sources, bool is_camera) {
        captures.resize(sources.size());
        current_frames.resize(sources.size());

        for (size_t i = 0; i < sources.size(); ++i) {
            if (is_camera) {
                int camera_id = stoi(sources[i]);
                if (!captures[i].open(camera_id)) {
                    cerr << "Failed to open camera " << camera_id << endl;
                    return false;
                }
            } else {
                if (!captures[i].open(sources[i])) {
                    cerr << "Failed to open video " << sources[i] << endl;
                    return false;
                }
            }
        }

        return true;
    }

    bool read(vector<Mat>& frames) {
        frames.resize(captures.size());

        for (size_t i = 0; i < captures.size(); ++i) {
            if (!captures[i].read(frames[i])) {
                return false;
            }
        }

        return true;
    }

    void release() {
        for (auto& cap : captures) {
            cap.release();
        }
    }

    Size getFrameSize() {
        if (!captures.empty() && captures[0].isOpened()) {
            return Size(captures[0].get(CV_CAP_PROP_FRAME_WIDTH),
                       captures[0].get(CV_CAP_PROP_FRAME_HEIGHT));
        }
        return Size(0, 0);
    }
};

// Main stitching pipeline
class RealtimeStitchingPipeline {
private:
    CachedStitcher stitcher;
    Config config;
    VideoSource video_source;
    VideoWriter output_writer;
    FPSCounter fps_counter;

public:
    RealtimeStitchingPipeline(const Config& cfg) : config(cfg) {
        // Create optimized stitcher
        stitcher = CachedStitcher::createOptimized(config.use_gpu);

        // Configure for real-time performance
        stitcher.setNumCudaStreams(4);
        stitcher.setUsePinnedMemory(true);
    }

    bool initialize() {
        // Open video sources
        if (config.use_camera || config.use_video) {
            if (!video_source.open(config.input_sources, config.use_camera)) {
                return false;
            }
        }

        // Setup output video writer if needed
        if (!config.output_file.empty()) {
            Size frame_size = getOutputSize();
            int fourcc = CV_FOURCC('M', 'J', 'P', 'G');
            output_writer.open(config.output_file, fourcc, config.target_fps,
                             frame_size, true);

            if (!output_writer.isOpened()) {
                cerr << "Failed to open output video file: " << config.output_file << endl;
                return false;
            }
        }

        return true;
    }

    bool calibrate() {
        cout << "Calibrating... ";
        cout.flush();

        vector<Mat> calibration_frames;

        // For static images
        if (!config.use_video && !config.use_camera) {
            for (const auto& source : config.input_sources) {
                Mat img = imread(source);
                if (img.empty()) {
                    cerr << "Failed to load image: " << source << endl;
                    return false;
                }
                calibration_frames.push_back(img);
            }
        }
        // For video/camera sources
        else {
            // Capture calibration frames
            for (int i = 0; i < config.calibration_frames; ++i) {
                vector<Mat> frames;
                if (!video_source.read(frames)) {
                    cerr << "Failed to capture calibration frames" << endl;
                    return false;
                }

                // Use first set of frames for calibration
                if (i == 0) {
                    calibration_frames = frames;
                }

                // Skip some frames to allow auto-exposure to settle
                if (i < 5) {
                    waitKey(100);
                }
            }
        }

        // Perform calibration
        auto start = chrono::high_resolution_clock::now();

        CachedStitcher::Status status = stitcher.cacheTransformations(calibration_frames);

        auto end = chrono::high_resolution_clock::now();
        double calibration_time = chrono::duration<double>(end - start).count();

        if (status != CachedStitcher::OK) {
            cerr << "Calibration failed!" << endl;
            return false;
        }

        cout << "Done! (" << calibration_time << " seconds)" << endl;

        // Display performance stats
        auto stats = stitcher.getPerformanceStats();
        cout << "GPU Memory Usage: " << stats.gpu_memory_mb << " MB" << endl;

        return true;
    }

    void run() {
        Mat panorama;
        bool running = true;

        fps_counter.start();

        // For static images, just process once
        if (!config.use_video && !config.use_camera) {
            vector<Mat> images;
            for (const auto& source : config.input_sources) {
                images.push_back(imread(source));
            }

            auto status = stitcher.composePanoramaGPU(images, panorama);
            if (status == CachedStitcher::OK) {
                imshow("Panorama", panorama);
                cout << "Press any key to exit..." << endl;
                waitKey(0);
            }
            return;
        }

        // Main processing loop for video/camera
        cout << "Processing... Press 'q' to quit" << endl;

        while (running) {
            vector<Mat> frames;

            // Capture frames
            if (!video_source.read(frames)) {
                cout << "End of video stream" << endl;
                break;
            }

            // Perform stitching
            auto status = stitcher.composePanoramaGPU(frames, panorama);

            if (status == CachedStitcher::OK) {
                // Update FPS counter
                fps_counter.tick();

                // Display FPS overlay
                if (config.show_fps) {
                    fps_counter.display(panorama);
                }

                // Display debug info
                if (config.debug_mode) {
                    displayDebugInfo(panorama);
                }

                // Show panorama
                imshow("Real-time Panorama", panorama);
                if (config.bench_mode) {
                    displayBenchInfo();
                }

                // Write to output file
                if (output_writer.isOpened()) {
                    output_writer.write(panorama);
                }

                // Check for quit
                int key = waitKey(1);
                if (key == 'q' || key == 27) {  // 'q' or ESC
                    running = false;
                } else if (key == 'd') {
                    config.debug_mode = !config.debug_mode;
                } else if (key == 'f') {
                    config.show_fps = !config.show_fps;
                }
            } else {
                cerr << "Stitching failed for frame" << endl;
            }
        }

        // Display final statistics
        displayFinalStats();
    }

    void cleanup() {
        video_source.release();
        if (output_writer.isOpened()) {
            output_writer.release();
        }
        destroyAllWindows();
    }

private:
    Size getOutputSize() {
        if (config.resolution == "720p") {
            return Size(1280, 720);
        } else if (config.resolution == "1080p") {
            return Size(1920, 1080);
        } else if (config.resolution == "4k") {
            return Size(3840, 2160);
        }

        // Default to input size * 2
        Size input_size = video_source.getFrameSize();
        return Size(input_size.width * 2, input_size.height);
    }

    void displayDebugInfo(Mat& image) {
        auto stats = stitcher.getPerformanceStats();

        stringstream ss;
        ss << "Compose: " << fixed << setprecision(1)
           << stats.last_compose_time_ms << " ms";
        putText(image, ss.str(), Point(10, 60), FONT_HERSHEY_SIMPLEX,
                0.7, Scalar(0, 255, 255), 2);

        ss.str("");
        ss << "Avg FPS: " << fixed << setprecision(1) << stats.avg_fps;
        putText(image, ss.str(), Point(10, 90), FONT_HERSHEY_SIMPLEX,
                0.7, Scalar(0, 255, 255), 2);
    }

    void displayFinalStats() {
        auto stats = stitcher.getPerformanceStats();

        cout << "\n=== Performance Statistics ===" << endl;
        cout << "Total frames processed: " << stats.frames_processed << endl;
        cout << "Average FPS: " << fixed << setprecision(2) << stats.avg_fps << endl;
        cout << "Average compose time: " << setprecision(2)
             << (1000.0 / stats.avg_fps) << " ms" << endl;
        cout << "GPU memory usage: " << stats.gpu_memory_mb << " MB" << endl;
        cout << "Calibration time: " << stats.transform_cache_time_ms << " ms" << endl;
    }

    void displayBenchInfo() {
        auto s = stitcher.getPerformanceStats();
        cout << fixed << setprecision(2)
             << "Upload: " << s.upload_time_ms << " ms, "
             << "Warp: " << s.warp_time_ms << " ms, "
             << "Exposure: " << s.exposure_time_ms << " ms, "
             << "Blend: " << s.blend_time_ms << " ms, "
             << "Download: " << s.download_time_ms << " ms" << endl;
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Real-time Image Stitching Demo" << endl;
        cout << "\nUsage:" << endl;
        cout << "  Static images:  " << argv[0] << " img1.jpg img2.jpg img3.jpg ..." << endl;
        cout << "  Video files:    " << argv[0] << " --video video1.mp4 video2.mp4 ..." << endl;
        cout << "  Webcams:        " << argv[0] << " --camera 0 1 2 ..." << endl;
        cout << "\nOptions:" << endl;
        cout << "  --gpu/--no-gpu  Enable/disable GPU acceleration" << endl;
        cout << "  --output FILE   Save output to video file" << endl;
        cout << "  --fps N         Target FPS (default: 30)" << endl;
        cout << "  --resolution    Output resolution (720p/1080p/4k)" << endl;
        cout << "  --calibrate N   Calibration frames (default: 10)" << endl;
        cout << "  --debug         Show debug information" << endl;
        return 1;
    }

    // Parse command-line arguments
    Config config = parseArgs(argc, argv);

    if (config.input_sources.empty()) {
        cerr << "No input sources specified!" << endl;
        return 1;
    }

    // Print configuration
    cout << "=== Configuration ===" << endl;
    cout << "GPU acceleration: " << (config.use_gpu ? "Enabled" : "Disabled") << endl;
    cout << "Input sources: " << config.input_sources.size() << endl;
    cout << "Target FPS: " << config.target_fps << endl;
    cout << "Resolution: " << config.resolution << endl;

    // Create and run pipeline
    RealtimeStitchingPipeline pipeline(config);

    if (!pipeline.initialize()) {
        cerr << "Failed to initialize pipeline!" << endl;
        return 1;
    }

    if (!pipeline.calibrate()) {
        cerr << "Calibration failed!" << endl;
        return 1;
    }

    pipeline.run();
    pipeline.cleanup();

    cout << "Done!" << endl;
    return 0;
}
