
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif

#include <iostream>
#include <chrono>
using namespace std;
using namespace cv;
using namespace cv::detail;
using namespace std::chrono; 


// Video parameters
int outputFPS = 60;
vector<string> video_names;
int readFrame = 0;
int startFrame = 0 ;
int stopFrame = 0;
bool isInitialized = false;
bool view_img = false;
int resizeWidth = 0;
string result_name = "result_video.mp4";
char key = 'k'; // Initialize quit key to a random char



Stitcher::Mode mode = Stitcher::PANORAMA;
vector<Mat> imgs;

// GPU/CUDA support
bool try_use_gpu = false; // OpenCV should be built with CUDa flags enabled. Removed in OpenCV 4.0 ?


// Registration working image size
double work_megapix = 0.6;

// Default command line args
Ptr<FeaturesMatcher> matcher;
string matcher_type = "homography";

// Features detector
// Initialize matcher confidence (ORB features defailt value is 0.3. Decrease this value if image has difficulty to stitch)
string features_type = "orb";
float match_conf = 0.3f;
Ptr<Feature2D> finder;

// Warper creator settings
Ptr<WarperCreator> warper;
string warper_type = "spherical";

// Blender settings
Ptr<Blender> blender;
int blender_type = Blender::MULTI_BAND;

// Seam finder settings
Ptr<SeamFinder> seam_finder;
string seam_finder_type = "gc_color";

// Seam estimation image resolution
double seam_megapix = 0.1;

// Exposure compensator
Ptr<ExposureCompensator> compensator;
string expos_comp_type = "gain_blocks";
int expos_comp = ExposureCompensator::GAIN_BLOCKS;



// Output return code information
string CODE_INFO = "[INFO] ";
string CODE_ERROR = "[ERROR] "; 



void printUsage(char** argv);
int parseCmdArgs(int argc, char** argv);
void cleanImgs();
void cleanStitcherParams();
void printSticherParams();
void initStartFrame(int &startFrame,VideoCapture &leftCapture,VideoCapture &rightCapture, Mat &leftFrame, Mat &rightFrame);



// The main code is a modified version of the original image sticher from OpenCV.
// The below modified code enables video support for stitching and relevant stitching' parameters can be tuned.
// https://github.com/opencv/opencv/blob/master/samples/cpp/stitching.cpp

int main(int argc, char* argv[])
{
    int retval = parseCmdArgs(argc, argv);
    if (retval) return EXIT_FAILURE;

    // Capture left and right videos
    if (video_names.empty())
        return EXIT_FAILURE;
    VideoCapture leftCapture(video_names[0]);
    VideoCapture rightCapture(video_names[1]);
    int leftTotalFrames = int(leftCapture.get(CAP_PROP_FRAME_COUNT));
    int rightTotalFrames = int(rightCapture.get(CAP_PROP_FRAME_COUNT));
    // Set the number of frames of the final panorama video
    if (leftTotalFrames != rightTotalFrames || stopFrame == 0)
    {
        stopFrame = min(leftTotalFrames, rightTotalFrames);
        cout << CODE_INFO << "Streams Total frames are different. stopFrame will be set to the minimum "<< stopFrame << endl;

    }
        
     

    // Initialize camera
    if(!leftCapture.isOpened() || !rightCapture.isOpened())
    {
        cout << "Error opening video stream" << endl; 
        return -1; 
    } 

    // Initialize Video writer object
    VideoWriter outputVideo; 

    // Initialize stitcher pointer, stitcher status and panorama output
    // source : https://github.com/opencv/opencv/blob/master/modules/stitching/src/stitcher.cpp
    Ptr<Stitcher> stitcher = Stitcher::create(mode);


    // Initialize work_megapix. Default is 0.6 Mpx.
    if (work_megapix != 0.6)
        stitcher->setRegistrationResol(work_megapix);
    
    // Initialize features detector. Default is ORB.
    if (features_type == "orb")
        finder = ORB::create();
    else if (features_type == "akaze")
        finder = AKAZE::create();
#ifdef HAVE_OPENCV_XFEATURES2D
    else if (features_type == "surf")
        finder = xfeatures2d::SURF::create();
#endif
    else if (features_type == "sift")
        finder = SIFT::create();
    else
    {
        cout << "Unknown 2D features type: '" << features_type << "'.\n";
        return -1;
    }
    stitcher->setFeaturesFinder(finder);

    // Initialize matcher type. Default is homography.
    if (matcher_type == "affine")
    {
        
        matcher = makePtr<AffineBestOf2NearestMatcher>(false, try_use_gpu, match_conf); // second argument is CUDA support
        stitcher->setFeaturesMatcher(matcher);

    }
    else if (matcher_type == "homography")
    {
        matcher = makePtr<BestOf2NearestMatcher>(try_use_gpu, match_conf); 
        stitcher->setFeaturesMatcher(matcher);
    }
    // Initialize warper type. Default is spherical
#ifdef HAVE_OPENCV_CUDAWARPING
    if (try_use_gpu)
    {
        cout << CODE_INFO <<  "CUDA GPU WARPING ..."<< endl;
        if (warper_type == "plane")
            warper = makePtr<cv::PlaneWarperGpu>();
        else if (warper_type == "cylindrical")
            warper = makePtr<cv::CylindricalWarperGpu>();
        else if (warper_type == "spherical")
            warper = makePtr<cv::SphericalWarperGpu>();
    }
    else
#endif
    {
        if (warper_type == "plane")
            warper = makePtr<cv::PlaneWarper>();
        else if (warper_type == "affine")
            warper = makePtr<cv::AffineWarper>();
        else if (warper_type == "cylindrical")
            warper = makePtr<cv::CylindricalWarper>();
        else if (warper_type == "spherical")
            warper = makePtr<cv::SphericalWarper>();
    }
    stitcher->setWarper(warper);
    

    // Initialize blender. Default is MultibandBlender

    if (blender_type == Blender::NO)
        blender = makePtr<detail::Blender>();
    else if (blender_type == Blender::FEATHER)
        blender = makePtr<detail::FeatherBlender>();
    else if (blender_type == Blender::MULTI_BAND)
        blender = makePtr<detail::MultiBandBlender>(try_use_gpu); // todo check params + check default param(nb of bands) of stitch / composepanorama
    stitcher->setBlender(blender);

    // Initialize seam_finder_type. Default is gc_color
    if (seam_finder_type == "no")
        seam_finder = makePtr<detail::NoSeamFinder>();
    else if (seam_finder_type == "voronoi")
        seam_finder = makePtr<detail::VoronoiSeamFinder>();

    else if (seam_finder_type == "gc_color")
    {
#ifdef HAVE_OPENCV_CUDALEGACY
    if (try_use_gpu)
        cout << CODE_INFO <<  "CUDA GPU SEAM FINDER ..."<< endl;
        seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR);
    else
#endif
        seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
    }
        
    else if (seam_finder_type == "gc_colorgrad")
    {
#ifdef HAVE_OPENCV_CUDALEGACY
    if (try_use_gpu)
        cout << CODE_INFO <<  "CUDA GPU SEAM FINDER ..."<< endl;
        seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
    else
#endif
        seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
    }
        
    else if (seam_finder_type == "dp_color")
        seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
    else if (seam_finder_type == "dp_colorgrad")
        seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);

    if (!seam_finder)
    {
        cout << CODE_INFO  <<"Can't create the following seam finder '" << seam_finder_type << "'\n";
        return 1;
    }
    stitcher->setSeamFinder(seam_finder);
    

    // Initialize seam_megapix. Default is 0.1 Mpx.
    if (seam_megapix != 0.1)
        stitcher->setSeamEstimationResol(seam_megapix);
    

    // Initialize Exposure compensator. Default is GAIN_BLOCKS.
    if (expos_comp != ExposureCompensator::GAIN_BLOCKS)
    {
        cout << CODE_INFO <<  "Exposure compensator set to "<< expos_comp << endl;
        stitcher->setExposureCompensator(compensator);
    }

    // Display sticher parameters
    printSticherParams();


    Stitcher::Status status; // stitching status
    Mat pano; // Panorama, stitched image

    // Initialize timers
    high_resolution_clock::time_point start = high_resolution_clock::now(); // Total execution time
    high_resolution_clock::time_point startTimeFrame; // one single frame processing time
    high_resolution_clock::time_point stopTimeFrame;

    // Loop through all the video stream
    while ( leftCapture.isOpened() && rightCapture.isOpened())
    {
        // Start timer
        startTimeFrame = high_resolution_clock::now(); 


        // Initialize left frame and right frame
        Mat leftFrame;
        Mat rightFrame;
        // Start the stitching on the given frame by providing the frame number
        if (startFrame != 0)
            initStartFrame(startFrame,leftCapture,rightCapture, leftFrame, rightFrame);


        // Read frame from left stream
        leftCapture >> leftFrame;
        if (leftFrame.empty())
        {
            cout << CODE_ERROR << "Can't read image left '"  << "'\n";
            break;
        }
        if(resizeWidth > 0)
            resize(leftFrame, leftFrame,Size(resizeWidth, (resizeWidth * leftFrame.size().height) / leftFrame.size().width),0,0, INTER_LINEAR);
        imgs.push_back(leftFrame);
        
        // Read frame from right stream
        rightCapture >> rightFrame;
        if (rightFrame.empty())
        {
            cout << CODE_ERROR << "Can't read image right'"  << "'\n";
            break;
        }
        if(resizeWidth > 0)
            resize(rightFrame, rightFrame,Size(resizeWidth, (resizeWidth * rightFrame.size().height) / rightFrame.size().width),0,0, INTER_LINEAR);
        

        imgs.push_back(rightFrame);

        // Estimate transform between the two cameras on the first frame
        if (readFrame == 0)
        {
            cout << CODE_INFO<< "Estimating the cameras' transform ..." << endl;
            status = stitcher->estimateTransform(imgs);

            if (status != Stitcher::OK)
            {
                // Clear current images vector and estimate the transform in the next frame if an error occured
                cout << CODE_ERROR << "Can't estimate the transform. "<< "Status : " << int(status) << endl;
                imgs.clear();
                continue;
            }
            else
            {
                // 
                std::vector<detail::CameraParams> cameras_ = stitcher->cameras();
                cout << CODE_INFO << "Number of cameras : " << cameras_.size() << endl;
                cout << CODE_INFO << "Transform successfully estimated on frame. Status : " << int(status) << endl;
                cout << CODE_INFO << "Cameras 0, K : \n " << cameras_[0].K() << endl;
                cout << CODE_INFO << "Cameras 0, R : \n " << cameras_[0].R << endl;
                cout << CODE_INFO << "Cameras 1, K : \n " << cameras_[1].K() << endl;
                cout << CODE_INFO << "Cameras 1, R : \n " << cameras_[1].R << endl; 
            }
            
        }
        

        // Stitching the two images into a panorama view
        status = stitcher->composePanorama(imgs, pano);
        if (status != Stitcher::OK)
        {
            cout << CODE_ERROR  << "Can't stitch images, error code = " << int(status) << endl;
            return EXIT_FAILURE;
        }
        


        // If the estimation of transform fails, the below code won't be executed.
        if (!isInitialized)
        {
            // Initialize video shape with pano output shape
            // fourcc : "x,x,x,x,"
            // ex : ('M', 'J', 'P','G')
            // TODO : if panorama size is too large, try next frames (in order to get better keypoints)
            if (pano.size().width < 8192 && pano.size().height < 8192)
                outputVideo.open(result_name,VideoWriter::fourcc('m','p','4','v'),outputFPS, Size(pano.size().width, pano.size().height)); // mp4v codec does not support video with size larger than 8192x8192.
            else
                outputVideo.open("result.avi",VideoWriter::fourcc('M','J','P','G'),40, Size(pano.size().width, pano.size().height)); // .AVI supports large video but the compression is very small 

            cout << CODE_INFO << "VideoWriter initialized (" << outputFPS << ") with the following shape : " <<  pano.size().width << "x" << pano.size().height << endl;
            isInitialized = true;
        }

        
        // Save frame in video
        outputVideo.write(pano); 

        // Calculate execution time for ONE single frame
        stopTimeFrame = high_resolution_clock::now(); 
        auto durationTimeFrame = duration_cast<microseconds>(stopTimeFrame - startTimeFrame);
        cout << CODE_INFO << "stitching frame : " << readFrame << "/" << stopFrame<< " completed successfully. " << durationTimeFrame.count() / 1000<< "ms." << endl;
        
        

        // Show stitching result on each frame. Quit Stitching by pressing 'q' 
        if (view_img) {
            // Show concatenated inputs
            Mat concatInputs;
            hconcat(leftFrame, rightFrame, concatInputs);
            resize(concatInputs, concatInputs, Size(1920, (1920 * concatInputs.size().height) / concatInputs.size().width),0,0,INTER_LINEAR);
            imshow("Concatenated Inputs", concatInputs);

            // Show stitched panorama
            resize(pano, pano,Size(1920, (1920 * pano.size().height) / pano.size().width),0,0, INTER_LINEAR);
            imshow("Stitched panorama", pano);

            key = (char) waitKey(1);
            if (key == 'q')
            {
                cout << CODE_INFO << "Successfully quit the program\n";
                cout << CODE_INFO << "Clean frame object" << endl;
                break;
            }
        }



        // Update reading state and clear images vector, frames Matrix
        readFrame++;
        imgs.clear();
        leftFrame.release();
        rightFrame.release();

        // Quit the reading if the stop frame is reached
        if (readFrame == stopFrame)
        {
           
            cout << CODE_INFO << "Stop frames reached." << endl;
            cout << CODE_INFO << "Clean frame object" << endl;
            leftFrame.release();
            rightFrame.release();
            break;
          
        } 
       
    }

    
    // Calculate total execution time
    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<microseconds>(stop - start); 
    cout << CODE_INFO << "Execution time  : " << duration.count() / 1000000 << " s" << endl; 
    cout << CODE_INFO << "Clean and deallocate memory" << endl;

    // Clean memory
    cleanStitcherParams();
    leftCapture.release();
    rightCapture.release();
    outputVideo.release();
    pano.release();
    cleanImgs();

    
    return EXIT_SUCCESS;
}


void printUsage(char** argv)
{
    cout <<
         "Video stitcher.\n\n" << "Usage :\n" << argv[0] <<" [Flags] video1 video2\n\n"
         "Flags:\n"
         "  --d3\n"
         "      internally creates three chunks of each image to increase stitching success\n"
         "  --mode (panorama|scans)\n"
         "      Determines configuration of stitcher. The default is 'panorama',\n"
         "      mode suitable for creating photo panoramas. Option 'scans' is suitable\n"
         "      for stitching materials under affine transformation, such as scans.\n"
         "  --build \n"
         "      Show the openCV build info'\n"
         "  --output <result_video>\n"
         "      The default is 'result.mp4'.\n"
         "      Try to use CUDA. The default value is 'no'. All default values\n"
         "      are for CPU mode.\n"
         "  --features (surf|orb|sift|akaze)\n"
         "      Type of features used for images matching.\n"
         "      The default is surf if available, orb otherwise.\n"
         "  --work_megapix <float>\n"
         "      Resolution for image registration step. The default is 0.6 Mpx.\n"
         "  --warp (affine|plane|cylindrical|spherical)\n"
         "      Warp surface type. The default is 'spherical'.\n"
         "  --matcher (homography|affine)\n"
         "      Matcher used for pairwise image matching.\n"
         "  --match_conf <float>\n"
         "      Confidence for feature matching step. The default is 0.65 for surf and 0.3 for orb.\n"
         "  --blend (no|feather|multiband) \n"
         "      The default blender is MultiBandBlender \n"
         "  --seam (no|voronoi|gc_color|gc_colorgrad)\n"
         "      Seam estimation method. The default is 'gc_color'.\n"
         "  --seam_megapix <float>\n"
         "      Resolution for seam estimation step. The default is 0.1 Mpx.\n"
         "  --expos_comp (no|gain|gain_blocks|channels|channels_blocks)\n"
         "      Exposure compensation method. The default is 'gain_blocks'.\n"
         "  --view-img \n"
         "      Show each frame stitching preview.\n"   
         "  --width <int>\n"
         "      Perfom stitching on a resized image (usually smaller for time performance and preview ).\n" 
         "  --stop <int>\n" 
         "      Stop the stitching at the given frame number.\n\n"
         "Example usage :\n" << argv[0] << " left_video.mp4 right_video.mp4 --output result.mp4 --matcher affine\n";
}



int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        printUsage(argv);
        return EXIT_FAILURE;
    }

    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            printUsage(argv);
            return EXIT_FAILURE;
        }

        else if (string(argv[i]) == "--output")
        {
            result_name = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--mode")
        {
            if (string(argv[i + 1]) == "panorama")
                mode = Stitcher::PANORAMA;
            else if (string(argv[i + 1]) == "scans")
                mode = Stitcher::SCANS;
            else
            {
                cout << "Bad --mode flag value\n";
                return EXIT_FAILURE;
            }
            i++;
        }
        else if (string(argv[i]) == "--build")
        {
            cout << getBuildInformation() << endl;
        }
        
        else if (string(argv[i]) == "--try_use_gpu")
        {
            
            try_use_gpu = true;
            
        }
        else if (string(argv[i]) == "--features")
        {
            features_type = argv[i + 1];
            if (string(features_type) == "orb")
                match_conf = 0.3f;
            else
                match_conf = 0.65f; // Default value for other features detector (SURF, SIFT ...)
            i++;
        }
        else if (string(argv[i]) == "--work_megapix")
        {
            work_megapix = atof(argv[i + 1]);
            i++;
        }

        else if (string(argv[i]) == "--seam_megapix")
        {
            seam_megapix = atof(argv[i + 1]);
            i++;
        }

        else if (string(argv[i]) == "--matcher")
        {
            if (string(argv[i + 1]) == "homography" || string(argv[i + 1]) == "affine")
                matcher_type = argv[i + 1];
            else
            {
                cout << "Bad --matcher flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--match_conf")
        {
            match_conf = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--blend")
        {
            if (string(argv[i + 1]) == "no")
                blender_type = Blender::NO;
            else if (string(argv[i + 1]) == "feather")
                blender_type = Blender::FEATHER;
            else if (string(argv[i + 1]) == "multiband")
                blender_type = Blender::MULTI_BAND;
            else
            {
                cout << CODE_INFO <<  "Bad blending method\n";
                return -1;
            }
            i++;
        }

        else if (string(argv[i]) == "--seam")
        {
            seam_finder_type = argv[i+1];
            i++;
        }
        else if (string(argv[i]) == "--seam_megapix")
        {
            seam_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--warp")
        {
            warper_type = argv[i+1];
            i++;
        }
        else if (string(argv[i]) == "--expos_comp")
        {
            expos_comp_type = string(argv[i + 1]);

            if (string(argv[i + 1]) == "no")
                expos_comp = ExposureCompensator::NO;
            else if (string(argv[i + 1]) == "gain")
                expos_comp = ExposureCompensator::GAIN;
            else if (string(argv[i + 1]) == "gain_blocks")
                expos_comp = ExposureCompensator::GAIN_BLOCKS;
            else if (string(argv[i + 1]) == "channels")
                expos_comp = ExposureCompensator::CHANNELS;
            else if (string(argv[i + 1]) == "channels_blocks")
                expos_comp = ExposureCompensator::CHANNELS_BLOCKS;
            else
            {
                cout << "Bad exposure compensation method\n";
                return -1;
            }

            compensator = ExposureCompensator::createDefault(expos_comp);
            i++;
        }
        else if (string(argv[i]) == "--stop")
        {
            stopFrame = stoi(argv[i+1]);
            i++;
        }

        else if (string(argv[i]) == "--start")
        {
            startFrame = stoi(argv[i+1]);
            i++;
        }


        else if (string(argv[i]) == "--fps")
        {
            outputFPS = stoi(argv[i+1]);
            i++;
        }

        else if (string(argv[i]) == "--view-img")
        {
            view_img = true;
        }

        else if (string(argv[i]) == "--width")
        {
            resizeWidth = stoi(argv[i+1]);
            i++;
        }

        

        else
        {
            video_names.push_back(argv[i]);

        }
    }
    return EXIT_SUCCESS;
}



void initStartFrame(int &startFrame,VideoCapture &leftCapture,VideoCapture &rightCapture, Mat &leftFrame, Mat &rightFrame)
{
    for (int i  = 0 ; i < startFrame ; i++)
    {
        leftCapture >> leftFrame;
        rightCapture >> rightFrame; 
        cout << CODE_INFO << " Init start frame ...  "<< i << "frames read" << endl;
    }
    cout << CODE_INFO << "Start frame initialized to "<< startFrame << endl;
    startFrame = 0;
    
}
void cleanImgs()
{
    for (int i = 0 ; i < imgs.size() ; i++)
    {
        imgs[i].release();
    }

}

void cleanStitcherParams()
{
    matcher.release();
    blender.release();
    warper.release();
    seam_finder.release();
    compensator.release();
    
}

void printSticherParams()
{
    cout << "  ------ Sticher parameters -----\n" << endl;
    cout << CODE_INFO << "CUDA GPU support set to " << try_use_gpu << endl;
    cout << CODE_INFO <<  "Registration resolution set to "<< work_megapix << endl;
    cout << CODE_INFO <<  "Features matcher set to  "<< features_type << endl;
    cout << CODE_INFO <<  "Matcher is set to "<< matcher_type << ", match_conf set to " << match_conf<< endl;
    cout << CODE_INFO <<  "Warper set to " << warper_type<< endl;
    cout << CODE_INFO <<  "Seam finder set to "<< seam_finder_type << endl;
    cout << CODE_INFO <<  "Seam estimation resolution set to "<< seam_megapix << endl;
    cout << CODE_INFO <<  "expos_comp_type set to "<< expos_comp_type << endl;
    if (blender_type == Blender::NO)
        cout << CODE_INFO <<  "Blender set to NO "<< endl; 
    else if (blender_type == Blender::FEATHER)
        cout << CODE_INFO <<  "Blender set to FEATHER "<< endl;
    else if (blender_type == Blender::MULTI_BAND)
        cout << CODE_INFO << "Blender set to Multiband "<< endl;
    cout << "  ---------------------------------\n" << endl;
    

}