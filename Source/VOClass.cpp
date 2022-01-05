#include "../Include/VOClass.h"
#include "../Include/Logger.h"
#include "../Include/Utils.h"
#include "../Include/Constants.h"
#include <fstream>

VOClass::VOClass(void){

}

VOClass::~VOClass(void){

}

void VOClass::constructProjectionMatrix(std::string line, cv::Mat& projectionMat){
    /* split line to words
    */
    std::vector<std::string> sub = tokenize(line);
    /* skip first word
    */
    int k = 1;
    for(int r = 0; r < 3; r++){
        for(int c = 0; c < 4; c++){
            projectionMat.at<float>(r, c) = std::stof(sub[k++]);
        }
    }
}

void VOClass::constructExtrinsicMatrix(std::string line){
    /* split line to words
    */
    std::vector<std::string> sub = tokenize(line);
    int k = 0;
    for(int r = 0; r < 3; r++){
        for(int c = 0; c < 4; c++){
            extrinsicMat.at<float>(r, c) = std::stof(sub[k++]);
        }
    }
    /* add last row since that is omitted in the text file
     * last row will be [0, 0, 0, 1]
    */
    extrinsicMat.at<float>(3, 3) = 1;
}

void VOClass::extractRT(cv::Mat& R, cv::Mat& T){
    /* extract 3x3 R
    */
    for(int r = 0; r < 3; r++){
        for(int c = 0; c < 3; c++){
            R.at<float>(r, c) = extrinsicMat.at<float>(r, c);
        }
    }
    /* extract 3x1 R
    */
    for(int r = 0; r < 3; r++)
        T.at<float>(r, 0) = extrinsicMat.at<float>(r, 3);
}

/* this function will be called in a loop to read through all sets of 
 * images within a sequence directory frameNumber will go from 1 to 
 * sizeof(imageDir)
*/
bool VOClass::readStereoImagesT1T2(int frameNumber){
    assert(frameNumber != 0);
    /* construct image file name from frameNumber, and img
     * format (.png in our case)
    */
    const int nameWidth = 6;
    /* read image pair at t = 1
    */
    std::string imgName = formatStringWidth(frameNumber - 1, nameWidth) + ".png";
    imgLT1 = cv::imread(leftImagesPath + imgName, cv::ImreadModes::IMREAD_GRAYSCALE);
    if(imgLT1.empty()){
        Logger.addLog(Logger.levels[ERROR], "Unable to open imgLT1", "leftImagesPath + imgName");
        assert(false);
    }

    imgRT1 = cv::imread(rightImagesPath + imgName, cv::ImreadModes::IMREAD_GRAYSCALE);
    if(imgRT1.empty()){
        Logger.addLog(Logger.levels[ERROR], "Unable to open imgRT1", "rightImagesPath + imgName");
        assert(false);        
    }

#if 0
    testShowStereoImage(imgLT1, imgRT1, frameNumber-1);
#endif

    /* read image pair at t = 2
    */
    imgName = formatStringWidth(frameNumber, nameWidth) + ".png";
    imgLT2 = cv::imread(leftImagesPath + imgName, cv::ImreadModes::IMREAD_GRAYSCALE);
    if(imgLT2.empty()){
        Logger.addLog(Logger.levels[ERROR], "Unable to open imgLT2", "leftImagesPath + imgName");
        assert(false);
    }

    imgRT2 = cv::imread(rightImagesPath + imgName, cv::ImreadModes::IMREAD_GRAYSCALE);
    if(imgRT2.empty()){
        Logger.addLog(Logger.levels[ERROR], "Unable to open imgRT2", "rightImagesPath + imgName");
        assert(false);        
    }

#if 0
    testShowStereoImage(imgLT2, imgRT2, frameNumber);
#endif
    return true;
}

/* calib.txt: Calibration data for the cameras
 * P0/P1 are the  3x4 projection matrices after rectification. Here P0 
 * denotes the left and P1 denotes the right camera. 
 * P2/P3 are left color camera and right color camera, which we won't be 
 * using here
 * 
 * It is basically the intrinsic matrix plus baseline component with camera 0 
 * (left) as reference
 *  _                        _
 *  | fu    0   cx  -fu * bx |
 *  | 0     fv  cy  0        |
 *  | 0     0   0   1        |
 *  -                        - 
*/
bool VOClass::getProjectionMatrices(const std::string calibrationFile){
    /* ifstream to read from file
    */
    std::ifstream file(calibrationFile);
    if(file.is_open()){
        std::string line;
        /* read first line
        */
        std::getline(file, line);
        constructProjectionMatrix(line, projectionCL);
        Logger.addLog(Logger.levels[INFO], "Constructed projectionCL");
        for(int r = 0; r < 3; r++){
            Logger.addLog(Logger.levels[DEBUG], projectionCL.at<float>(r, 0), 
                                                projectionCL.at<float>(r, 1), 
                                                projectionCL.at<float>(r, 2), 
                                                projectionCL.at<float>(r, 3)
            );
        }
        /* read second line
        */
        std::getline(file, line);
        constructProjectionMatrix(line, projectionCR);
        Logger.addLog(Logger.levels[INFO], "Constructed projectionCR");
        for(int r = 0; r < 3; r++){
            Logger.addLog(Logger.levels[DEBUG], projectionCR.at<float>(r, 0), 
                                                projectionCR.at<float>(r, 1), 
                                                projectionCR.at<float>(r, 2), 
                                                projectionCR.at<float>(r, 3)
            );
        }
        return true;  
    }
    else{
        Logger.addLog(Logger.levels[ERROR], "Unable to open calibration file");
        assert(false);
    }
}

/* poses/XX.txt contains the 4x4 homogeneous matrix flattened out
 * to 12 elements; r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
 *  _                _
 *  | r11 r12 r13 tx |
 *  | r21 r22 r21 ty |
 *  | r31 r32 r33 tz |
 *  | 0   0   0   1  |
 *  -                - 
*/
bool VOClass::getGroundTruthPath(const std::string groundTruthFile){
    /* ifstream to read from file
    */
    std::ifstream file(groundTruthFile);
    if(file.is_open()){
        std::string line;
        /* read all lines
        */
        while(std::getline(file, line)){
            constructExtrinsicMatrix(line);
            /* extract R, T from extrinsicMat
            */
            cv::Mat R = cv::Mat::zeros(3, 3, CV_32F);
            cv::Mat T = cv::Mat::zeros(3, 1, CV_32F);
            extractRT(R, T);
            /* construct ground x, y, z
             * If you are interested in getting where camera0 is located in the 
             * world coordinate frame, you can transform the origin (0,0,0) of the 
             * camera0's local coordinate frame to the world coordinate
             * 
             * ? = R * {0, 0, 0} + T
             * This tells where the camera is located in the world coordinate.
            */
            groundX.push_back(T.at<float>(0, 0));
            groundY.push_back(T.at<float>(1, 0));
            groundZ.push_back(T.at<float>(2, 0));
#if 0
            /* display one instance of the extrinsic matrix
            */
            Logger.addLog(Logger.levels[DEBUG], "Constructed extrinsicMat");
            for(int r = 0; r < 4; r++){
                Logger.addLog(Logger.levels[DEBUG], extrinsicMat.at<float>(r, 0), 
                                                    extrinsicMat.at<float>(r, 1), 
                                                    extrinsicMat.at<float>(r, 2), 
                                                    extrinsicMat.at<float>(r, 3)
                );
            }
            /* display one instance of R
            */
            Logger.addLog(Logger.levels[DEBUG], "Extracted R from extrinsicMat");
            for(int r = 0; r < 3; r++){
                Logger.addLog(Logger.levels[DEBUG], R.at<float>(r, 0), 
                                                    R.at<float>(r, 1), 
                                                    R.at<float>(r, 2)
                );
            }
            /* display one instance of T
            */
            Logger.addLog(Logger.levels[DEBUG], "Extracted T from extrinsicMat");
            for(int r = 0; r < 3; r++){
                Logger.addLog(Logger.levels[DEBUG], T.at<float>(r, 0)
                );
            }
            /* display ground x, y, z
            */
            Logger.addLog(Logger.levels[DEBUG], "Computed groundX, groundY, groundZ");
            Logger.addLog(Logger.levels[DEBUG], T.at<float>(0, 0), 
                                                T.at<float>(1, 0), 
                                                T.at<float>(2, 0));
#endif
        }
        Logger.addLog(Logger.levels[INFO], "Constructed ground truth trajectory", groundX.size());
#if 0
        testShowGroundTruthTrajectory();
#endif
        return true;
    }
    else{
        Logger.addLog(Logger.levels[ERROR], "Unable to open ground truth file");
        assert(false);
    }
}

/* compute disparity map using sgbm method (semi global
 * matching); Not going into too much detail, this algorithm
 * matches blocks instead of pixels
*/
cv::Mat VOClass::computeDisparity(cv::Mat leftImg, cv::Mat rightImg){
    /* tunable parameters
     *
     * Minimum possible disparity value. Normally, it is zero but 
     * sometimes rectification algorithms can shift images, so this 
     * parameter needs to be adjusted accordingly.
    */
    int minDisparity = 0;
    /* Maximum disparity minus minimum disparity. The value is always 
     * greater than zero. It must be divisible by 16 (implementation
     * requirement)
    */
    int numDisparities = 32;
    /* Matched block size. It must be an odd number >=1 . Normally, it 
     * should be somewhere in the 3..11 range.
    */
    int blockSize = 5;
    /* Parameters controlling the disparity smoothness, the larger the 
     * values are, the smoother the disparity is. P2 must be > P1. The
     * recommended values for P1 = 8 * numChannels * blockSize^2, 
     * P2 = 32 * numChannels * blockSize^2 
    */
    int P1 = 8 * 1 * blockSize * blockSize;
    int P2 = 32 * 1 * blockSize * blockSize;

    /* create block matching object
    */
    cv::Ptr<cv::StereoSGBM> pStereo = cv::StereoSGBM::create(minDisparity = minDisparity, 
                                                             numDisparities = numDisparities, 
                                                             blockSize = blockSize, 
                                                             P1 = P1, 
                                                             P2 = P2
    ); 
    /* compute disparity map
    */
    cv::Mat disparityMap;
    pStereo->compute(leftImg, rightImg, disparityMap);
    Logger.addLog(Logger.levels[INFO], "Computed disparity map");

#if 0
    testShowDisparityImage(leftImg, rightImg, disparityMap);
#endif
    return disparityMap;
}

/* FAST feature detection for detecting corners
*/
std::vector<cv::Point2f> VOClass::getFeaturesFAST(cv::Mat img){
    /* The keypoint is characterized by the 2D position, scale (proportional 
     * to the diameter of the neighborhood that needs to be taken into account), 
     * orientation and some other parameters. 
     * 
     * The keypoint neighborhood is then analyzed by another algorithm that builds 
     * a descriptor (usually represented as a feature vector)
    */
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::Point2f> featurePoints;
    /* theshold: threshold on difference between intensity of the central pixel 
     * and pixels of a circle around this pixel. 
     * pixel p is a corner if there exists a set of n contiguous pixels in the 
     * circle (of 16 pixels) which are all brighter than Ip+t, or all darker than 
     * Ip−t.
     * 
     * nonmaxSuppression: algorithm faces issues when there are adjacent keypoints,
     * so a score matrix is computed and the one with the lower value is discarded
     * https://docs.opencv.org/4.x/df/d0c/tutorial_py_fast.html
    */
    int threshold = 20;
    bool nonmaxSuppression = true;
    cv::FAST(img, keypoints, threshold, nonmaxSuppression);  
    /* This method converts vector of keypoints to vector of points
    */
    cv::KeyPoint::convert(keypoints, featurePoints);
    Logger.addLog(Logger.levels[INFO], "Computed feature vector", featurePoints.size());

#if 0
    testShowDetectedFeatures(img, featurePoints);
#endif
    return featurePoints;
}

/* KLT feature matcher based on Sparse Optical Flow
 * NOTE: There are 2 types of optical flow. Dense and sparse. Dense finds 
 * flow for all the pixels while sparse finds flow for the selected points.
 * 
 * Match feature from imgLT1 to LT2, remove all other points from feature
 * vector
*/
void VOClass::matchFeatureKLT(std::vector<cv::Point2f>& featurePointsLT1){
    /* create termination criteria for optical flow calculation
     * The first argument of this function tells the algorithm that we want 
     * to terminate either after some number of iterations or when the 
     * convergence metric reaches some small value (respectively). The next 
     * two arguments set the values at which one, the other, or both of these 
     * criteria should terminate the algorithm.
     * 
     * The reason we have both options is so we can  stop when either limit is 
     * reached.
     * 
     * Here, the criteria specifies the termination criteria of the iterative 
     * search algorithm (after the specified maximum number of iterations maxCount 
     * or when the search window moves by less than epsilon) in the pyramid.
    */
    const int maxCount = 30;
    const float epsilon = 0.01;
    cv::TermCriteria termCrit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 
                                                 maxCount, epsilon);
    
    /* circular matching of features using optical flow
     * (1) using optical flow, calculate and match imgLT1 features in imgRT1
     * (2) using the imgRT1 matched feature from above, match and find features
     *     in imgRT2 using optical flow
     * (3) similarly, find and match features in imgLT2 using imgRT2 features
     * (4) Finally, find, match and store features in imgLT1 using imgLT2 features
     * Now, you will have two sets of feature vectors for imgLT1
    */
    std::vector<cv::Point2f> featurePointsRT1, featurePointsRT2, featurePointsLT2;
    std::vector<cv::Point2f> featurePointsLT1Re;
    /* output status vector (of unsigned chars); each element of the vector is set
     * to 1 if the flow for the corresponding features has been found, otherwise, 
     * it is set to 0. We uses these to remove unmatched features from the vector
    */
    std::vector<unsigned char> status0, status1, status2, status3;
    /* err:  output vector of errors; each element of the vector is set to an error 
     * for the corresponding feature. Optical flow basically works by matching a patch, 
     * around each input point, from the input image to the second image. The parameter 
     * err allows you to retrieve the matching error (e.g. you may think of that as the 
     * correlation error) for each input point.
     * 
     * winSize: size of the search window at each pyramid level.
     * 
     * pyramidLevels: if set to 0, pyramids are not used (single level), if set to 1, 
     * two levels are used, and so on.
    */
    std::vector<float> err;                    
    cv::Size winSize = cv::Size(21,21); 
    const int pyramidLevels = 3;

    cv::calcOpticalFlowPyrLK(imgLT1, imgRT1, featurePointsLT1, featurePointsRT1, 
                             status0, err, winSize, pyramidLevels, termCrit);
    cv::calcOpticalFlowPyrLK(imgRT1, imgRT2, featurePointsRT1, featurePointsRT2, 
                             status1, err, winSize, pyramidLevels, termCrit);
    cv::calcOpticalFlowPyrLK(imgRT2, imgLT2, featurePointsRT2, featurePointsLT2, 
                             status2, err, winSize, pyramidLevels, termCrit);
    cv::calcOpticalFlowPyrLK(imgLT2, imgLT1, featurePointsLT2, featurePointsLT1Re, 
                             status3, err, winSize, pyramidLevels, termCrit);

    Logger.addLog(Logger.levels[INFO], "Circular matching complete");
    Logger.addLog(Logger.levels[INFO], "Output status vector sizes", 
    status0.size(), status1.size(), status2.size(), status3.size());

#if 1
    
#endif
    /* delete unmatched feature points; status vector element would
     * be 0, or calculated point (x,y) would be out of bounds
    */
}

/* this displays the stereo pair side by side
*/
void VOClass::testShowStereoImage(cv::Mat imgLeft, cv::Mat imgRight, int frameNumber){
    Logger.addLog(Logger.levels[TEST], "Show stereo pair images", frameNumber);

    cv::Mat imgPair;
    cv::vconcat(imgLeft, imgRight, imgPair);

    /* Mat, text, point, font, font scale, color, thickness
    */
    std::string text = "FRAME: " + std::to_string(frameNumber);
    cv::putText(imgPair, text, cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 
                3, cv::Scalar(255, 255, 255), 3
    );
    imshow("Stereo Pair", imgPair);
    cv::waitKey(0);
}

/* read stereo pair, compute disparity and display them
*/
void VOClass::testShowDisparityImage(cv::Mat imgLeft, cv::Mat imgRight, cv::Mat disparityMap){
    Logger.addLog(Logger.levels[TEST], "Show disparity and stereo pair images");

    cv::Mat imgPair, imgTuple;
    cv::vconcat(imgLeft, imgRight, imgPair);

    /* disparityMap type is of type CV_16S and imgPair is of type CV_8U, to
     * concat, both of them need to be of the same type
    */
    cv::Mat disparityMapTypeChanged;
    disparityMap.convertTo(disparityMapTypeChanged, CV_8UC1);

    cv::vconcat(imgPair, disparityMapTypeChanged, imgTuple);
    imshow("Disparity Map", imgTuple);
    cv::waitKey(0);
}

/* plot ground truth trajectory
*/
void VOClass::testShowGroundTruthTrajectory(void){
    int numPoints = groundX.size();
    Logger.addLog(Logger.levels[TEST], "Show ground truth trajectory");
    /* create an empty image
    */
    const int trajectoryR = 800;
    const int trajectoryC = 800;
    cv::Mat trajectory = cv::Mat::zeros(trajectoryR, trajectoryC, CV_8UC3);
    for(int i = 0; i < numPoints; i++){
        /* shift origin of the path so that the entire path is visible, 
         * default origin is at top left of the screen
         *
         * NOTE:
         * Some trajectories may go out of bounds and become invisible,
         * the origin shift has to be updated accordingly
         * o---------------------> x or cols
         * |
         * |
         * |
         * | y or rows
         * v
        */

        /* the camera on the car is facing the z axis, so to get a
         * top down view, we plot x-z axis
        */
        int p1 = groundX[i] + trajectoryC/2;
        int p2 = groundZ[i] + trajectoryR/4;
        /* img, center, radius, color, thickness
         */
        /* different color for the starting point and ending point
         */
        if(i == 0)
            cv::circle(trajectory, cv::Point(p1, p2), 5, CV_RGB(0, 255, 0), 2);
        else if(i == numPoints - 1)
            cv::circle(trajectory, cv::Point(p1, p2), 5, CV_RGB(255, 0, 0), 2);
        else
            cv::circle(trajectory, cv::Point(p1, p2), 1, CV_RGB(255, 255, 0), 2);   
    }

    imshow("Ground Truth", trajectory);
    cv::waitKey(0); 
}

/* display detected features
*/
void VOClass::testShowDetectedFeatures(cv::Mat img, std::vector<cv::Point2f> featurePoints){
    Logger.addLog(Logger.levels[TEST], "Show features detected");
    /* img is a single channel image, we will convert it to 3
     * to color our feature points
    */
    cv::Mat imgChannelChanged;
    cv::cvtColor(img, imgChannelChanged, cv::COLOR_GRAY2RGB);
    /* mark all feature  points on img
    */
    int numFeatures = featurePoints.size();
    for(int i = 0; i < numFeatures; i++){
        cv::circle(imgChannelChanged, featurePoints[i], 1, CV_RGB(0, 0, 255), 2);
    }
    imshow("Feature Points", imgChannelChanged);
    cv::waitKey(0);
}