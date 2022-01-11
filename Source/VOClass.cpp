#include "../Include/VOClass.h"
#include "../Include/Logger.h"
#include "../Include/Utils.h"
#include "../Include/Constants.h"
#include <cmath>
#include <fstream>

VOClass::VOClass(void){
    /* width = number of cols (x)
     * height = number of rows (y)
    */
    frameW = 1241; 
    frameH = 376;
}

VOClass::~VOClass(void){
}

/* this function will be called in a loop to read through all sets of images 
 * within  a sequence directory frameNumber will go from 0 to sizeof(imageDir)-1
*/
bool VOClass::readStereoImagesT1T2(int frameNumber){
    /* construct image file name from frameNumber, and img
     * format (.png in our case)
    */
    const int nameWidth = 6;
    /* read image pair at t = 1
    */
    std::string imgName = formatStringWidth(frameNumber, nameWidth) + ".png";
    imgLT1 = cv::imread(leftImagesPath + imgName, cv::ImreadModes::IMREAD_GRAYSCALE);
    Logger.addLog(Logger.levels[INFO], "Read image", leftImagesPath + imgName, imgLT1.rows, 
                                                                               imgLT1.cols);
    if(imgLT1.empty()){
        Logger.addLog(Logger.levels[ERROR], "Unable to open imgLT1", "leftImagesPath + imgName");
        assert(false);
    }

    imgRT1 = cv::imread(rightImagesPath + imgName, cv::ImreadModes::IMREAD_GRAYSCALE);
    Logger.addLog(Logger.levels[INFO], "Read image", rightImagesPath + imgName, imgRT1.rows, 
                                                                                imgRT1.cols);
    if(imgRT1.empty()){
        Logger.addLog(Logger.levels[ERROR], "Unable to open imgRT1", "rightImagesPath + imgName");
        assert(false);        
    }

    /* read image pair at t+1
    */
    imgName = formatStringWidth(frameNumber+1, nameWidth) + ".png";
    imgLT2 = cv::imread(leftImagesPath + imgName, cv::ImreadModes::IMREAD_GRAYSCALE);
    Logger.addLog(Logger.levels[INFO], "Read image", leftImagesPath + imgName, imgLT2.rows, 
                                                                               imgLT2.cols);
    if(imgLT2.empty()){
        Logger.addLog(Logger.levels[ERROR], "Unable to open imgLT2", "leftImagesPath + imgName");
        assert(false);
    }

    imgRT2 = cv::imread(rightImagesPath + imgName, cv::ImreadModes::IMREAD_GRAYSCALE);
    Logger.addLog(Logger.levels[INFO], "Read image", rightImagesPath + imgName, imgRT2.rows, 
                                                                                imgRT2.cols);
    if(imgRT2.empty()){
        Logger.addLog(Logger.levels[ERROR], "Unable to open imgRT2", "rightImagesPath + imgName");
        assert(false);        
    }

#if SHOW_STEREO_IMAGE_PAIR
    testShowStereoImage(imgLT1, imgRT1, frameNumber);
    testShowStereoImage(imgLT2, imgRT2, frameNumber+1);
#endif
    return true;
}

/* calib.txt: Calibration data for the cameras
 * P0/P1 are the  3x4 projection matrices after rectification. Here P0 denotes the 
 * left and P1 denotes the right camera. P2/P3 are left color camera and right color 
 * camera, which we won't be using here
 * 
 * These matrices contain intrinsic information about the camera's focal length and 
 * optical center. Further, they also contain tranformation information which relates 
 * each camera's coordinate frame to the global coordinate frame (in this case that 
 * of the left grayscale camera). - rectified projection matrix
 * 
 * The global frame is the established coordinate frame of the camera's first position
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

/* poses/XX.txt contains the 4x4 homogeneous matrix flattened out to 12 elements; 
 * r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
 *  _                _
 *  | r11 r12 r13 tx |
 *  | r21 r22 r21 ty |
 *  | r31 r32 r33 tz |
 *  | 0   0   0   1  |
 *  -                - 
 * 
 * The number 12 comes from flattening a 3x4 transformation matrix of the left 
 * stereo camera with respect to the global coordinate frame (first frame of
 * left camera)
*/
bool VOClass::getGroundTruthPath(const std::string groundTruthFile, int& numFrames){
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
             * ? = R * {0, 0, 0} + T; or the same calculation in homogeneous coords,
             * [R|t] * [0, 0, 0, 1]
             * 
             * This tells where the camera is located in the world coordinate.
             * The resulting (x, y, z) will be in meteres
            */
            groundTruth.push_back(T);
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
            Logger.addLog(Logger.levels[DEBUG], "Computed groundTruth");
            Logger.addLog(Logger.levels[DEBUG], T.at<float>(0, 0), 
                                                T.at<float>(1, 0), 
                                                T.at<float>(2, 0));
#endif
        }
#if SHOW_GROUND_TRUTH_TRAJECTORY
        testShowGroundTruthTrajectory();
#endif
        numFrames = groundTruth.size();
        Logger.addLog(Logger.levels[INFO], "Constructed ground truth trajectory", numFrames);
        return true;
    }
    else{
        Logger.addLog(Logger.levels[ERROR], "Unable to open ground truth file");
        assert(false);
    }
}

/* compute disparity map using sgbm method (semi global matching); Not going 
 * into too much detail, this algorithm matches blocks instead of pixels
*/
cv::Mat VOClass::computeDisparity(cv::Mat leftImg, cv::Mat rightImg){
    /* tunable parameters
     *
     * Minimum possible disparity value. Normally, it is zero but 
     * sometimes rectification algorithms can shift images, so this 
     * parameter needs to be adjusted accordingly.
    */
    int _minDisparity = 0;
    /* Maximum disparity minus minimum disparity. The value is always 
     * greater than zero. It must be divisible by 16 (implementation
     * requirement)
    */
    int _numDisparities = 96;
    /* Matched block size. It must be an odd number >=1 . Normally, it 
     * should be somewhere in the 3..11 range.
    */
    int _blockSize = 11;
    /* Parameters controlling the disparity smoothness, the larger the 
     * values are, the smoother the disparity is. P2 must be > P1. The
     * recommended values for P1 = 8 * numChannels * blockSize^2, 
     * P2 = 32 * numChannels * blockSize^2 
    */
    int _P1 = 8 * 1 * _blockSize * _blockSize;
    int _P2 = 32 * 1 * _blockSize * _blockSize;

    /* create block matching object
    */
    cv::Ptr<cv::StereoSGBM> pStereo = cv::StereoSGBM::create(0);
    pStereo->setMinDisparity(_minDisparity);
    pStereo->setNumDisparities(_numDisparities);
    pStereo->setBlockSize(_blockSize);
    pStereo->setP1(_P1);
    pStereo->setP2(_P2);
    pStereo->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

    /* compute disparity map
    */
    cv::Mat disparityMap;
    /* Output disparity map has the same size as the input images. When 
     * disptype==CV_16S, the map is a 16-bit signed single-channel image, 
     * containing disparity values scaled by 16. To get the true disparity 
     * values from such fixed-point representation, you will need to divide 
     * each disp element by 16. So if you've chosen disptype = CV_16S during 
     * computation, you can access a pixel at pixel-position (X,Y) by
     * 
     * short pixVal = disparity.at<short>(Y,X);
     * while the disparity value is
     * float disparity = pixVal / 16.0f;
     * 
     * If disptype==CV_32F, the disparity map will already contain the real 
     * disparity values on output. if you've chosen disptype = CV_32F during 
     * computation, you can access the disparity directly:
     * 
     * float disparity = disparity.at<float>(Y,X)
    */
    pStereo->compute(leftImg, rightImg, disparityMap);
    Logger.addLog(Logger.levels[INFO], "Computed disparity map", disparityMap.type());

    /* we will use the true disparity map for point cloud computation, since
     * displaying this would not yield anything good
     * NOTE: We can see that there is a gap of the left side of the image, this is 
     * because the right camera did not have matching information. 
     * The disparity value here would be -1.0
    */
    cv::Mat trueDisparityMap;
    disparityMap.convertTo(trueDisparityMap, CV_32F, 1.0f/16.0f);
    Logger.addLog(Logger.levels[INFO], "Computed true disparity map", trueDisparityMap.type());

#if SHOW_DISPARITY_MAP
    cv::Mat disparityMap8Bit;
    trueDisparityMap.convertTo(disparityMap8Bit, CV_8U);
    testShowDisparityImage(leftImg, rightImg, disparityMap8Bit);
#endif
    return trueDisparityMap;
}

/* convert disparity map to depth map using f, Tx, d
 * NOTE: the depthMap points are NOT to be confused with actual 3D points
 * in camera frame !!! To get 3D points, we will need to invert the intrinsic
 * matrix to get x, y in camera frame
*/
cv::Mat VOClass::computeDepthMap(cv::Mat disparityMap){
    /* compute fx from intrinsic matrix (for left camera)
    */
    float focalLengthX = projectionCL.at<float>(0, 0);
    /* compute Tx (baseline)
    */
    float baseline = (projectionCR.at<float>(0, 3))/ (-1 * focalLengthX);
    /* avoid division by 0 since the disparityMap might have 0.0 or -1.0 values; 
     * we change these values to 0.1 which would result in large depth. we can
     * filter these out after 
    */
    for(int r = 0; r < frameH; r++){
        for(int c = 0; c < frameW; c++){
            float d = disparityMap.at<float>(r, c);
            if(d == 0.0 || d == -1.0)
                disparityMap.at<float>(r, c) = 0.1;
        }
    }
    /* create an empty depth map with same shape as disparityMap
    */
    float maxDepth = 0, minDepth = INT_MAX;
    cv::Mat depthMap = cv::Mat::ones(disparityMap.rows, disparityMap.cols, CV_32F);
    for(int r = 0; r < frameH; r++){
        for(int c = 0; c < frameW; c++){
            float d = disparityMap.at<float>(r, c);
            depthMap.at<float>(r, c) = focalLengthX * baseline/d;   
            /* compute max depth and min depth; for info purpose
            */   
            maxDepth = std::max(depthMap.at<float>(r, c), maxDepth);
            minDepth = std::min(depthMap.at<float>(r, c), minDepth);
        }
    }
    Logger.addLog(Logger.levels[INFO], "Computed depth map", focalLengthX, baseline, 
                                                             minDepth, maxDepth);
#if WRITE_DEPTH_PLY_FILE
    cv::Mat colors;
    /* use imgLT1 if we are passing in disparityMapT1, and imgLT2 if
     * passing disparityMapT2
    */
    cv::cvtColor(imgLT1, colors, cv::COLOR_GRAY2RGB);

    int *hist = computeHistogram(depthMap, maxDepth);
    /* subtract elements with maxDepth since we will be filtering them
     * from vertices list while writing to file
    */
    int numVertices = (depthMap.rows * depthMap.cols) - hist[(int)maxDepth];
    writeToPLY(depthMap, colors, 3000, numVertices);
    free(hist);
#endif

#if SHOW_DEPTH_MAP
    testShowDepthImage(disparityMap, depthMap);
#endif
    return depthMap;
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

#if SHOW_ALL_FAST_FEATURES
    testShowDetectedFeatures(img, featurePoints);
#endif
    return featurePoints;
}

/* KLT feature matcher based on Sparse Optical Flow
 * NOTE: There are 2 types of optical flow. Dense and sparse. Dense finds 
 * flow for all the pixels while sparse finds flow for the selected points.
 * 
 * The circular algorithm takes a feature from one frame and finds the best 
 * match in an another frame, sequentially following the order Left(t) - Right(t) 
 * - Right(t+1)-Left(t+1)-Left(t)
 * 
 * Finally, if the feature is successfully matched through the entire sequence, 
 * i.e., the last matched feature is the same as the beginning one, the circle 
 * is closed and the feature is considered as `being stable` and therefore kept 
 * as a high-quality point for further analysis
*/
std::vector<cv::Point2f> VOClass::matchFeatureKLT(std::vector<cv::Point2f> &featurePointsLT1){
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
    const int maxCount = 50;
    const float epsilon = 0.03;
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
    cv::Size winSize = cv::Size(15,15); 
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
    Logger.addLog(Logger.levels[INFO], "Feature vector sizes", 
    featurePointsLT1.size(), featurePointsRT1.size(), 
    featurePointsRT2.size(), featurePointsLT2.size(), featurePointsLT1Re.size());

    Logger.addLog(Logger.levels[INFO], "Status vector sizes", 
    status0.size(), status1.size(), status2.size(), status3.size());

    Logger.addLog(Logger.levels[INFO], "Status vector valid points", 
    validMatches(status0), validMatches(status1), validMatches(status2), validMatches(status3));   

#if SHOW_CIRCULAR_MATCHING_PAIR
    testShowCirculatMatchingPair(imgLT1, featurePointsLT1, featurePointsRT1, status0);
    testShowCirculatMatchingPair(imgRT1, featurePointsRT1, featurePointsRT2, status1);
    testShowCirculatMatchingPair(imgRT2, featurePointsRT2, featurePointsLT2, status2);
    testShowCirculatMatchingPair(imgLT2, featurePointsLT2, featurePointsLT1Re, status3);
#endif
    /* update status vector for invalid feature points; calculated point 
     * (x,y) would be out of bounds
    */
    markInvalidFeaturesBounds(featurePointsRT1, status0);
    markInvalidFeaturesBounds(featurePointsRT2, status1);
    markInvalidFeaturesBounds(featurePointsLT2, status2);
    markInvalidFeaturesBounds(featurePointsLT1Re, status3);

    Logger.addLog(Logger.levels[INFO], "Status vector valid points", "Bounds filter", 
    validMatches(status0), validMatches(status1), validMatches(status2), validMatches(status3));
 
#if SHOW_CIRCULAR_MATCHING_PAIR_BOUNDS_FILTER
    testShowCirculatMatchingPair(imgLT1, featurePointsLT1, featurePointsRT1, status0);
    testShowCirculatMatchingPair(imgRT1, featurePointsRT1, featurePointsRT2, status1);
    testShowCirculatMatchingPair(imgRT2, featurePointsRT2, featurePointsLT2, status2);
    testShowCirculatMatchingPair(imgLT2, featurePointsLT2, featurePointsLT1Re, status3);
#endif 
    /* extract common features across 4 images
    */
    std::vector<cv::Point2f> fLT1, fRT1, fRT2, fLT2, fLT1Re;
    for(int i = 0; i < status0.size(); i++){
        if(status0[i] == 1 && status1[i] == 1 && status2[i] == 1 && status3[i] == 1){
            fLT1.push_back(featurePointsLT1[i]);
            fRT1.push_back(featurePointsRT1[i]);
            fRT2.push_back(featurePointsRT2[i]);
            fLT2.push_back(featurePointsLT2[i]);
            fLT1Re.push_back(featurePointsLT1Re[i]);
        }
    }
    int numCommonFeatures = fLT1.size();
    Logger.addLog(Logger.levels[INFO], "Extracted common features", numCommonFeatures);

#if SHOW_CIRCULAR_MATCHING_QUAD
    testShowCirculatMatchingFull(fLT1, fRT1, fRT2, fLT2, fLT1Re);
#endif

    /* Loop closure; if fLT1 and fLT1Re are the same points then
     * those features are stable
    */
    std::vector<cv::Point2f> fLT1ReOffset, flT2Offset;
    int threshold = 2;
    for(int i = 0; i < numCommonFeatures; i++){
        int offset = std::max(std::abs(fLT1[i].x - fLT1Re[i].x), 
                              std::abs(fLT1[i].y - fLT1Re[i].y));

        if(offset < threshold){
            fLT1ReOffset.push_back(fLT1Re[i]);
            /* use the same features in fLT2, this is to maintain a 
             * 1:1 correspondence
            */
            flT2Offset.push_back(fLT2[i]);
        }
    }
    Logger.addLog(Logger.levels[INFO], "Extracted stable features", fLT1ReOffset.size(),
                                                                    flT2Offset.size());
#if 0
    for(int i = 0; i < fLT1ReOffset.size(); i++)
        Logger.addLog(Logger.levels[DEBUG], fLT1ReOffset[i].x, fLT1ReOffset[i].y, 
                                            flT2Offset[i].x, flT2Offset[i].y);
#endif

#if SHOW_ALL_FAST_FEATURES_STABLE
    testShowDetectedFeatures(imgLT1, fLT1ReOffset);
    testShowDetectedFeatures(imgLT2, flT2Offset);
#endif
    /* return the final output after removing invalid features
     * the matching features between LT1 and LT2 are fLT1Re and fLT2
    */
    featurePointsLT1 = fLT1ReOffset;
    return flT2Offset;
}

/* estimate motion using matched feature points between LT1 and LT2
*/
cv::Mat VOClass::estimateMotion(std::vector<cv::Point2f> featurePointsT1, 
                                std::vector<cv::Point2f> featurePointsT2, 
                                cv::Mat depthMap){
    /*
    * First, we convert feature points (in image frame) to 3D points in camera 
    * frame for LEFT IMAGE (global frame)
    *
    * [u, v, 1] = intrinsicMat * [x, y, z, 1]
    *    z.u    = fx  0  cx 0      x
    *    z.v      0   fy cy 0  (x) y  
    *     z       0   0  0  0      z
    *                              1
    * 
    * We have already computed depth from depthMap using disparityMap
    * (u)z = fx(x) + (cx)z
    *  x = (u(z) - cx(z))/fx
    * Instead of taking the inverse of k, we compute x, y arithmetically
    */
    /* feature points whose depth are valid
    */
    std::vector<cv::Point2f> imagePointsT1, imagePointsT2;
    /* camera frame 3D points from imgLT1
    */
    std::vector<cv::Point3f> objectPoints;
    /* the matched array of features have to be the same size
    */
    assert(featurePointsT1.size() == featurePointsT2.size());
    int numFeatures =featurePointsT1.size();
    /* depth threshold
    */
    int depthThresh = 3000;
    /* extract cx, cy, fx, fy from prjectionMat of left camera
    */
    float cx = projectionCL.at<float>(0, 2);
    float cy = projectionCL.at<float>(1, 2);
    float fx = projectionCL.at<float>(0, 0);
    float fy = projectionCL.at<float>(1, 1);
    Logger.addLog(Logger.levels[INFO], "Extracted params from projectionCL", cx, cy, fx, fy);

    for(int i = 0; i < numFeatures; i++){
        /* (u, v) in image frame
        */
        float u = featurePointsT1[i].x;
        float v = featurePointsT1[i].y;
        /* compute depth of this point
         * NOTE: x goes in horizontal direction -> equivalent to cols
         *       y goes in vertical direction   -> equivalent to rows
        */
        float z = depthMap.at<float>(v, u);
        /* If the depth at the position of our matched feature is above threshold, 
         * then we ignore this feature because we don't actually know the depth and 
         * it will throw our calculations off
        */
        if(z > depthThresh)
            continue;
        /* filtered feature points
        */
        imagePointsT1.push_back({u, v});
        imagePointsT2.push_back({featurePointsT2[i].x, featurePointsT2[i].y});
        /* Use arithmetic to extract x and y (faster than using inverse of k)
        */
        float x = z * (u - cx)/fx;
        float y = z * (v - cy)/fy;
        /* store (x, y, z) these are in camera frame
        */
        objectPoints.push_back({x, y, z});
    }
    Logger.addLog(Logger.levels[INFO], "Feature points size before depth filter", 
    featurePointsT1.size());
    Logger.addLog(Logger.levels[INFO], "Feature points size after depth filter", 
    imagePointsT1.size());
    Logger.addLog(Logger.levels[INFO], "Object points size", objectPoints.size());
#if 0
    Logger.addLog(Logger.levels[DEBUG], "3D object points(T1) and 2D image points(T2)");
    for(int i = 0; i < objectPoints.size(); i++){
        Logger.addLog(Logger.levels[DEBUG], objectPoints[i].x, 
                                            objectPoints[i].y, 
                                            objectPoints[i].z,
                                            imagePointsT2[i].x,
                                            imagePointsT2[i].y);
    }
#endif

    /* output R matrix and T vector combined to form 4x4 matrix with last row
     * [0, 0, 0, 1]
    */
    cv::Mat Rt = cv::Mat::zeros(4, 4, CV_32F);
    cv::Mat R, t;
    /* output trajectory (x, y, z) at current instant
    */
    cv::Mat tPose = cv::Mat::zeros(3, 1, CV_32F);
    /* construct intrinsic matrix
    */
    cv::Mat K = cv::Mat::zeros(3, 3, CV_32F);
    for(int r = 0; r < 3; r++){
        for(int c = 0; c < 3; c++)
            K.at<float>(r, c) = projectionCL.at<float>(r, c);
    }
    /* Second, Pose estimation step;
     * We need to compute a pose that relates points in the global coordinate 
     * frame to the camera's pose. 
     * 
     * The global frame is the established coordinate frame of the camera's first 
     * position. If we look at the first line in pose.txt, we see that the
     * rotational component is identity, with a translation vector equal to zero 
     * for all axes.
     *  
     * We used the camera's pose in the first image (the first image in left 
     * camera is the origin in world frame) as the global coordinate frame, 
     * reconstruct 3D positions of the features in the image using stereo depth 
     * estimation, then find a pose (R|t matrix) which relates the camera in the 
     * next frame to those 3D points. This is exactly what the matrix in pose
     * file represents.
     * 
     *                /---/
     *               / x / Cl1
     *              /---/
     *              
     *  x x x x        R|t to transform 3D points in Cl1 frame to global frame (Cl0)
     *  _______
     * |  Cl0  |
     *  -------
     * When tracking the vehicle pose over time, what we actually want is to relate 
     * the points in the camera's coordinate frame to the global frame
    */
#if POSE_ESTIMATION_RANSAC
    /* RANSAC method
     * Using RANSAC is useful when you suspect that a few data points are extremely 
     * noisy. For example, consider the problem of fitting a line to 2D points. 
     * This problem can be solved using linear least squares where the distance of 
     * all points from the fitted line is minimized. Now consider one bad data point 
     * that is wildly off. This one data point can dominate the least squares solution 
     * and our estimate of the line would be very wrong. In RANSAC, the parameters are 
     * estimated by randomly selecting the minimum number of points required. In a line
     * fitting problem, we randomly select two points from all data and find the line 
     * passing through them. Other data points that are close enough to the line are 
     * called inliers. Several estimates of the line are obtained by randomly selecting 
     * two points, and the line with the maximum number of inliers is chosen as the 
     * correct estimate.
    */
    /* Rodrigues parameters are also called axis-angle rotation. They are formed by 4 
    * numbers [theta, x, y, z], which means that you have to rotate an angle "theta" 
    * around the axis described by unit vector v=[x, y, z]. But, in cv::Rodrigues 
    * function reference, it seems that OpenCV uses a "compact" representation of 
    * Rodrigues notation as vector with 3 elements rod2=[a, b, c], where:
    * 
    * Angle to rotate theta is the module of input vector 
    * theta = sqrt(a^2 + b^2 + c^2)
    * 
    * Rotation axis v is the normalized input vector: 
    * v = rod2/theta = [a/theta, b/theta, c/theta]
    */
    cv::Mat rRodrigues;
    /* Assuming no lens distortion
    */
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_32F);
    /* We actually want to relate the points in the camera's coordinate frame to the 
     * global frame, so we want the opposite (inverse) of the transformation matrix 
     * provided to us by the SolvePnPRansac function.
    */
    cv::solvePnPRansac(objectPoints, imagePointsT2, K , distCoeffs, rRodrigues, t);

    Logger.addLog(Logger.levels[INFO], "Estimated rRodrigues vector", 
                                        rRodrigues.rows, rRodrigues.cols);
    for(int r = 0; r < 3; r++)
        Logger.addLog(Logger.levels[DEBUG], rRodrigues.at<float>(r, 0));

    Logger.addLog(Logger.levels[INFO], "Estimated translation vector", 
                                        t.rows, t.cols);
    for(int r = 0; r < 3; r++)
        Logger.addLog(Logger.levels[DEBUG], t.at<float>(r, 0));

    /* convert rodrigues rotation vector to Euler angles notation, which represent 
     * three consecutive rotations around a combination of X, Y and Z axes.
    */
    cv::Rodrigues(rRodrigues, R);

    Logger.addLog(Logger.levels[INFO], "Estimated rotation vector", 
                                        R.rows, R.cols);
    for(int r = 0; r < 3; r++)
        Logger.addLog(Logger.levels[DEBUG], R.at<float>(r, 0), 
                                            R.at<float>(r, 1), 
                                            R.at<float>(r, 2));
    /* Combine R and t to Rt
    */
    for(int r = 0; r < 3; r++){
        for(int c = 0; c < 3; c++){
            Rt.at<float>(r, c) = R.at<float>(r, c); 
        }
        /* add last column
        */
        Rt.at<float>(r, 3) = t.at<float>(r, 0);
    }
    /* add last row [0, 0, 0, 1]
    */
    Rt.at<float>(3, 3) = 1;

    Logger.addLog(Logger.levels[INFO], "Estimated pose matrix", 
                                        Rt.rows, Rt.cols);
    for(int r = 0; r < 4; r++)
        Logger.addLog(Logger.levels[DEBUG], Rt.at<float>(r, 0), 
                                            Rt.at<float>(r, 1), 
                                            Rt.at<float>(r, 2),
                                            Rt.at<float>(r, 3));   
    
    /* Integrate all pose matrices
     * We are tracking the vehicle motion from the very first camera pose, so we 
     * need the cumulative product of the inverses of each estimated camera pose 
     * given to us by SolvePnPRansac.
     * 
     * This way, the transformation matrix at each index will be one that relates 
     * 3D homogeneous coordinates in the camera's frame to the global coordinate
     * frame, which is the coordinate frame of the camera's first position. The 
     * translation vector component of this transformation matrix will describe 
     * where the camera's curent origin is in this global referece frame.
    */
    poseRt = poseRt * (Rt.inv());
    Logger.addLog(Logger.levels[INFO], "Integrated pose matrix", 
                                        poseRt.rows, poseRt.cols);
    for(int r = 0; r < 4; r++)
        Logger.addLog(Logger.levels[DEBUG], poseRt.at<float>(r, 0), 
                                            poseRt.at<float>(r, 1), 
                                            poseRt.at<float>(r, 2),
                                            poseRt.at<float>(r, 3));  
    /* output pose (x, y, z) is the translational component of poseRt
    */
    tPose.at<float>(0, 0) = poseRt.at<float>(0, 3);
    tPose.at<float>(1, 0) = poseRt.at<float>(1, 3);
    tPose.at<float>(2, 0) = poseRt.at<float>(2, 3);
#endif

    Logger.addLog(Logger.levels[INFO], "Computed tPose");
    Logger.addLog(Logger.levels[INFO], tPose.at<float>(0, 0), 
                                       tPose.at<float>(1, 0),
                                       tPose.at<float>(2, 0));
    return tPose;
}

float VOClass::computeErrorInPoseEstimation(std::vector<cv::Mat> trajectory){
    Logger.addLog(Logger.levels[INFO], "Trajectory vector size", trajectory.size());
    Logger.addLog(Logger.levels[INFO], "Ground truth vector size", groundTruth.size());
    
    float error;
#if 1
    for(int i = 0; i < trajectory.size(); i++)
        Logger.addLog(Logger.levels[DEBUG], "Calculated: ", trajectory[i].at<float>(0, 0),
                                                            trajectory[i].at<float>(1, 0),
                                                            trajectory[i].at<float>(2, 0), 
                                            " Truth: ",     groundTruth[i].at<float>(0, 0), 
                                                            groundTruth[i].at<float>(1, 0),
                                                            groundTruth[i].at<float>(2, 0));
#endif
    
    return error;
}