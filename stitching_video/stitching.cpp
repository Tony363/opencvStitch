
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/stitching/detail/blenders.hpp"

#include <iostream>
#include <chrono>
using namespace std;
using namespace cv;
using namespace cv::detail;
using namespace std::chrono; 
#include <chrono>
bool divide_images = false;
bool try_use_gpu = false; // OpenCV should be built with CUDa flags enabled. Removed in OpenCV 4.0 ?
Stitcher::Mode mode = Stitcher::PANORAMA;
vector<Mat> imgs;

string result_name = "result_video.mp4";
char key = 'k';

// Video parameters
int outputWidth = 6000;
int outputHeight = 3000;
int outputFPS = 60;
vector<string> video_names;
int readFrame = 0;
int startFrame = 0 ;
int stopFrame = 0;
bool isInitialized = false;
bool show = false;
int resizeWidth = 0;

// Default command line args
Ptr<FeaturesMatcher> matcher;
string matcher_type = "homography";

// Warper creator settings
Ptr<WarperCreator> warper;
string warper_type = "";

// Blender settings
Ptr<Blender> blender;
string blender_type = "";

// Seam finder settings
Ptr<SeamFinder> seam_finder;
string seam_finder_type = "";

// Seam estimation image resolution
double seam_megapix = 0.1;

//int blender = Blender::FEATHER;

// output code information
string CODE_INFO = "[INFO] ";
string CODE_ERROR = "[ERROR] "; 



void printUsage(char** argv);
int parseCmdArgs(int argc, char** argv);
void cleanImgs();
void cleanStitcherParams();
void initStartFrame(int &startFrame,VideoCapture &leftCapture,VideoCapture &rightCapture, Mat &leftFrame, Mat &rightFrame);



// NB : modified version for videos
// Please check original source code to get images stitcher instead of the below video stitcher

int main(int argc, char* argv[])
{
    int retval = parseCmdArgs(argc, argv);
    if (retval) return EXIT_FAILURE;

    // Capture left and right videos
    VideoCapture leftCapture(video_names[0]);
    VideoCapture rightCapture(video_names[1]);
    int leftTotalFrames = int(leftCapture.get(CAP_PROP_FRAME_COUNT));
    int rightTotalFrames = int(rightCapture.get(CAP_PROP_FRAME_COUNT));
    if (leftTotalFrames != rightTotalFrames)
        cout << CODE_INFO << "Streams Total frames are different. Stopframe will be set to the minimum" << endl;
        
    if (stopFrame == 0)
        stopFrame = min(leftTotalFrames, rightTotalFrames);
    


    // Start timer
    auto start = high_resolution_clock::now(); 


    // Initialize camera
    if(!leftCapture.isOpened() || !rightCapture.isOpened())
    {
        cout << "Error opening video stream" << endl; 
        return -1; 
    } 

    // Initialize Video writer object
    VideoWriter outputVideo; 

    // Initialize stitcher pointer, stitcher status and panorama output
    Ptr<Stitcher> stitcher = Stitcher::create(mode);

    // Initialize matcher type. Default is homography.
    if (matcher_type == "affine")
    {
        cout << CODE_INFO <<  " Matcher is set to : affine"<< endl;
        matcher = makePtr<AffineBestOf2NearestMatcher>();
        stitcher->setFeaturesMatcher(matcher);

    }
    // Initialize warper type. Default is spherical
    if (warper_type != "")
    {
        cout << CODE_INFO <<  " warp set to " << warper_type<< endl;
        if (warper_type == "plane")
            warper = makePtr<cv::PlaneWarper>();
        else if (warper_type == "affine")
            warper = makePtr<cv::AffineWarper>();
        else if (warper_type == "cylindrical")
            warper = makePtr<cv::CylindricalWarper>();
        else if (warper_type == "spherical")
            warper = makePtr<cv::SphericalWarper>();

        stitcher->setWarper(warper);
    }

    // Initialize blender. Default is MultibandBlender
    if (blender_type != "")
    {
        cout << CODE_INFO <<  " Blender set to FEATHER "<< endl;
        blender = makePtr<detail::FeatherBlender>();
        stitcher->setBlender(blender);
    }

    // Initialize seam_finder_type. Default is gc_color
    if (seam_finder_type != "")
    {
        cout << CODE_INFO <<  " Seam Finder set to gc_grad"<< endl;
        seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
        stitcher->setSeamFinder(seam_finder);
    }

    // Initialize seam_megapix. Default is 0.1 Mpx.
    if (seam_megapix != 0.1)
    {
        cout << CODE_INFO <<  " Seam estimation resolution set to "<< seam_megapix << endl;
        stitcher->setSeamEstimationResol(seam_megapix);
    }

    Stitcher::Status status;
    Mat pano;

    high_resolution_clock::time_point startTimeFrame;
    high_resolution_clock::time_point stopTimeFrame;

    // Loop through all the video stream
    while ( leftCapture.isOpened() && rightCapture.isOpened())
    {
        // Frame processing starting time
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

        if (readFrame == 0)
        {
            status = stitcher->estimateTransform(imgs);

            if (status != Stitcher::OK)
            {
                // Clear current images vector and estimate the transform in the next frame
                cout << CODE_ERROR << "Can't estimate the transform. "<< "Status : " << int(status) << endl;
                imgs.clear();
                continue;
            }
            else
            {
                cout << CODE_INFO << "Transform successfully estimated on frame. Status : " << int(status) << endl;
            }
            
        }
        


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
                //outputVideo.open("result.h264",VideoWriter::fourcc('H','2','6','4'),outputFPS, Size(pano.size().width, pano.size().height));
                outputVideo.open("result.avi",VideoWriter::fourcc('M','J','P','G'),40, Size(pano.size().width, pano.size().height)); // Not ok with 60 fps

            cout << CODE_INFO << "VideoWriter initialized (" << outputFPS << ") with the following shape : " <<  pano.size().width << "x" << pano.size().height << endl;
            isInitialized = true;
        }

        
        // Save frame in video
        outputVideo.write(pano); 

        // End timer
        stopTimeFrame = high_resolution_clock::now(); 
        auto durationTimeFrame = duration_cast<microseconds>(stopTimeFrame - startTimeFrame);
        cout << CODE_INFO << "stitching frame : " << readFrame << "/" << stopFrame<< " completed successfully. " << durationTimeFrame.count() / 1000<< "ms." << endl;
        
        // Update reading state and clear images vector, frames Matrices
        readFrame++;
        imgs.clear();
        leftFrame.release();
        rightFrame.release();

        // Show stitching result on each frame. Quit Stitching by pressing 'q' 
        if (show) {
            
            imshow("Frame", pano);
            key = (char) waitKey(1);

            if (key == 'q')
            {
                cout << CODE_INFO << "Successfully quit the program\n";
                cout << CODE_INFO << "Clean frame object" << endl;
                leftFrame.release();
                rightFrame.release();
                break;
            }
        }


        if (readFrame == stopFrame)
        {
           
            cout << CODE_INFO << "Stop frames reached." << endl;
            cout << CODE_INFO << "Clean frame object" << endl;
            leftFrame.release();
            rightFrame.release();
            break;
          
        } 
       
    }

    
    // Stop time
    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<microseconds>(stop - start); 
    cout << CODE_INFO << "Execution time  : " << duration.count() / 1000000 << " s" << endl; 
    cout << CODE_INFO << "Clean and deallocate memory" << endl;
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
         "  --output <result_video>\n"
         "      The default is 'result.mp4'.\n"
         "  --warp (affine|plane|cylindrical|spherical)\n"
         "      Warp surface type. The default is 'spherical'.\n"
         "  --matcher (homography|affine)\n"
         "      Matcher used for pairwise image matching.\n"
         "  --blend (no|feather|multiband) \n"
         "      The default blender is MultiBandBlender \n"
         "  --seam (no|voronoi|gc_color|gc_colorgrad)\n"
         "      Seam estimation method. The default is 'gc_color'.\n"
         "  --seam_megapix <float>\n"
         "      Resolution for seam estimation step. The default is 0.1 Mpx.\n"
         "  --show\n"
         "      Show each frame stitching preview.\n"   
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
        else if (string(argv[i]) == "--d3")
        {
            divide_images = true;
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
        
        else if (string(argv[i]) == "--try_use_gpu")
        {
            if (string(argv[i + 1]) == "no")
                try_use_gpu = false;
            else if (string(argv[i + 1]) == "yes")
                try_use_gpu = true;
            else
            {
                cout << "Bad --try_use_gpu flag value\n";
                return EXIT_FAILURE;
            }
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

        else if (string(argv[i]) == "--blend")
        {
            blender_type = argv[i+1];
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

        else if (string(argv[i]) == "--show")
        {
            show = true;
        }

        else if (string(argv[i]) == "--width")
        {
            resizeWidth = stoi(argv[i+1]);
            i++;
        }

        

        else
        {
            video_names.push_back(argv[i]);
            /*
            Mat img = imread(samples::findFile(argv[i]));
            if (img.empty())
            {
                cout << "Can't read image '" << argv[i] << "'\n";
                return EXIT_FAILURE;
            }

            if (divide_images)
            {
                Rect rect(0, 0, img.cols / 2, img.rows);
                imgs.push_back(img(rect).clone());
                rect.x = img.cols / 3;
                imgs.push_back(img(rect).clone());
                rect.x = img.cols / 2;
                imgs.push_back(img(rect).clone());
            }
            else
                imgs.push_back(img);

            */
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
    
}