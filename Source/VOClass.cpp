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

/* this function will be called in a loop to read through all sets of images within 
 * a sequence directory frameNumber will go from 0 to sizeof(imageDir)-1
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
    testShowStereoImage(imgLT1, imgRT1, frameNumber);
#endif

    /* read image pair at t+1
    */
    imgName = formatStringWidth(frameNumber+1, nameWidth) + ".png";
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
    testShowStereoImage(imgLT2, imgRT2, frameNumber+1);
#endif
    return true;
}

/* calib.txt: Calibration data for the cameras
 * P0/P1 are the  3x4 projection matrices after rectification. Here P0 denotes the 
 * left and P1 denotes the right camera. P2/P3 are left color camera and right color 
 * camera, which we won't be using here
 * 
 * It is basically the intrinsic matrix plus baseline component with camera 0 (left) 
 * as reference
 *  _                        _
 *  | fu    0   cx  -fu * bx |
 *  | 0     fv  cy  0        |
 *  | 0     0   0   1        |
 *  -                        -
 * 
 * NOTE: Normally, a camera's projection matrix take a 3D point in a global 
 * coordinate frame and projects it onto pixel coordinates on THAT camera's 
 * image frame. Rectified projection matrices are the opposite, and are designed 
 * to map points in each camera's coordinate frame onto one single image plane: 
 * that of the left camera. This means they are going in the opposite direction, 
 * as these matrices are taking 3D points from the coordinate frame of the camera 
 * they are associated with, and projecting them onto the image plane of the left 
 * camera. 
 * 
 * NOTE: If the projection matrices provided were standard per-camera projection 
 * matrices, we would expect the extrinsic matrix to take a point from the global 
 * coordinate frame and tranform it into the frame of the camera. For example, if 
 * we were to take the origin of the global coordinate frame (the origin of the 
 * left grayscale camera) and translate it into the coordinate frame of the right 
 * grayscale camera, we would expect to see an X coordinate of -0.54, since the 
 * left camera's origin is 0.54 meters to the left of the right grayscale camera's 
 * origin.
 * 
 * To test this, we can decompose the projection matrix given for the right camera 
 * into k, R, and t.
 * Intrinsic Matrix:
 * [[718.856    0.     607.1928]
 * [  0.     718.856  185.2157]
 * [  0.       0.       1.    ]]
 * Rotation Matrix:
 * [[1. 0. 0.]
 * [0. 1. 0.]
 * [0. 0. 1.]]
 * Translation Vector:
 * [[ 0.5372]
 * [ 0.    ]
 * [0.     ]
 * [ 1.    ]]
 * 
 * In this case, let's see what we get if we transform the origin of the global 
 * coordinate frame (the origin of the left grayscale camera) using the tranformation/
 * extrinsic matrix we got from this projection matrix. According to the schematic, 
 * we should expect it to be 0.54m to the left (X = -0.54)
 * But, we get
 * [[ 0.5372]
 * [ 0.    ]
 * [0.     ]
 * 
 * And here we can see the issue: the X value is POSITIVE 0.54, which points 0.54m 
 * to the right of the camera frame, which tells us that this projection matrix is 
 * NOT actually referring to the right camera, it is referring to the left camera, 
 * and treating the origin of the right camera as the global coordinate frame.
 * 
 * This is because it is a rectified projection matrix for a stereo rig, which is 
 * intended to project points from the coordinate frames of multiple cameras onto 
 * the SAME image plane, rather than the coordinates from one coordinate frame onto
 * the image planes of multiple cameras.
 * 
 * Each rectified projection matrix will take (X, Y, Z, 1) homogeneous coordinates of 
 * 3D points in the associated sensor's coordinate frame and translates them to pixel 
 * locations (u, v, 1) on the image plane of the left grayscale camera.
 * 
 * Now we know what these calibration matrices are actually telling us, and that they 
 * are in all in reference to the left grayscale camera's image plane
 * 
 * EXERCISE: Let's take some point measured with respect to the coordinate frame of the 
 * left camera when it is in it's 14 pose of the sequence, and transform it onto the 
 * image plane of the camera
 * Original point:
 * [[1]
 * [2]
 * [3]
 * [1]]
 * 
 * We can do thisby using accessing the 14th pose from the ground truth (extrinsic 
 * matrix) and multiplying them
 * 
 * [R|t] * original point = [u, v, w] in camera frame
 * 'w' is the depth from camera
 * 
 * Transformed point:
 * [[ 0.2706]
 * [ 1.5461]
 * [15.0755]]
 * Depth from camera:
 * [15.0755]
 * 
 * To project this 3D point onto the image plane, we could first apply the intrinsic 
 * matrix, THEN divide by the depth, which would take us from meters, to pixel*meters, 
 * then to pixels
 * OR
 * or we could just divide by the depth first, taking us into unitless measurements 
 * by dividing meters by meters,then multiply by the intrinsic matrix, which would then 
 * take us from unitless to pixels
 * [[620.09802465 258.93763336   1.        ]]
 * 
 * Another thing that is done regularly is "normalizing" the pixel coordinates by 
 * multiplying them by the inverse of the intrinsic matrix k, which brings us back into 
 * unitless values, as we take pixel measurements and multiply them by the inverse of 
 * pixel measurements.
 * (which represent the 3D points projected onto a 2D plane which is 1 unit (unitless) 
 * away from the origin of the camera down the Z axis)
 * Normalized Coordinates: [[0.01795245 0.10255452 1.        ]]
 * 
 * it is interesting to see that if we take the normalized coordinates and multiply 
 * them by their respective depths, we can reconstruct the original 3D position from
 * the 2D projection of a point (as long as we know the depth)
 * [[ 0.2706,  1.5461, 15.0755]]
 * 
 * And we are back to 3D coordinates in our camera frame. To get back to the 3D position 
 * in our original coordinate frame, we can add a row of (0, 0, 0, 1) to the transformation 
 * matrix (R|t) we used to make it homogeneous, then invert it, and finally dot it with the 
 * restored 3D coordinates in the camera frame (note that we also need to add a 1 to the end 
 * of these coordinates to make them homogeneous as well) 
 * ([1., 2., 3., 1.])
 * 
 * And there we have it, we went from a 3D point on a coordinate frame, projected it to 
 * pixel coordinates of a camera in a separate coordinate frame, reconstructed the metrics 
 * using a known depth, and then reverse transformed the 3D point back into the original 
 * frame by inverting a homogeneous version of the original transformation matrix used and 
 * dotting it with the homogeneous 3D coordinates of the point in the camera's coordinate 
 * frame.
 * 
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

/* this matrix is to be used to convert to feature points to 3d world
 * coordinates
 * Tx (baseline) is in meters and all others are in pixels
 *  _                        _
 *  | 1    0    0      -cx    |
 *  | 0    1    0      -cy    |
 *  | 0    0    0      f      |
 *  | 0    0   -1/Tx   0      |
 *  -                        - 
*/
void VOClass::getQMatrix(void){
    float cx = projectionCL.at<float>(0, 2);
    float cy = projectionCL.at<float>(1, 2);
    /* fu and fv are the same here
    */
    float f  = projectionCL.at<float>(0, 0);
    float tx = (projectionCR.at<float>(0, 3))/(-1 * f);

    float data[16] = {1,    0,    0,    -cx, 
                      0,    1,    0,    -cy,
                      0,    0,    0,     f,
                      0,    0, -(1/tx),  0 
                      }; 
    for(int r = 0; r < 4; r++){
        for(int c = 0; c < 4; c++){
            qMat.at<float>(r, c) = data[r*4 + c];
        }
    }

    Logger.addLog(Logger.levels[INFO], "Constructed qMat");
    for(int r = 0; r < 4; r++){
        Logger.addLog(Logger.levels[DEBUG], qMat.at<float>(r, 0), 
                                            qMat.at<float>(r, 1), 
                                            qMat.at<float>(r, 2), 
                                            qMat.at<float>(r, 3));
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
             * ? = R * {0, 0, 0} + T; or the same calculation in homogeneous coords,
             * [R|t] * [0, 0, 0, 1]
             * 
             * This tells where the camera is located in the world coordinate.
             * The resulting (x, y, z) will be in meteres
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
    int _minDisparity = 0;
    /* Maximum disparity minus minimum disparity. The value is always 
     * greater than zero. It must be divisible by 16 (implementation
     * requirement)
    */
    int _numDisparities = 96;
    /* Matched block size. It must be an odd number >=1 . Normally, it 
     * should be somewhere in the 3..11 range.
    */
    int _blockSize = 18;
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

#if 1
    cv::Mat disparityMap8Bit;
    trueDisparityMap.convertTo(disparityMap8Bit, CV_8U);
    testShowDisparityImage(leftImg, rightImg, disparityMap8Bit);
#endif
    return trueDisparityMap;
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

#if 0
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
 
#if 0
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
#if 0
    testShowCirculatMatchingFull(fLT1, fRT1, fRT2, fLT2, fLT1Re);
#endif

    /* Loop closure; if fLT1 and fLT1Re are the same points then
     * those features are stable
    */
    std::vector<cv::Point2f> fLT1ReOffset, flT2Offset;
    int threshold = 5;
    for(int i = 0; i < numCommonFeatures; i++){
        int offset = std::max(std::abs(fLT1[i].x - fLT1Re[i].x), 
                              std::abs(fLT1[i].y - fLT1Re[i].y));

        if(offset < threshold){
            /* round fLT1Re to int
            */
            int x = round(fLT1Re[i].x);
            int y = round(fLT1Re[i].y);
            fLT1ReOffset.push_back(cv::Point2f(x, y));
            /* use the same features in fLT2, this is to maintain a 
             * 1:1 correspondence
            */
            flT2Offset.push_back(fLT2[i]);
#if 0
        Logger.addLog(Logger.levels[DEBUG], fLT1[i].x, fLT1Re[i].x, fLT1[i].y, fLT1Re[i].y);
#endif
        }
    }
    Logger.addLog(Logger.levels[INFO], "Extracted stable features", fLT1ReOffset.size(),
                                                                    flT2Offset.size());
#if 0
    testShowDetectedFeatures(imgLT1, fLT1ReOffset);
    testShowDetectedFeatures(imgLT2, flT2Offset);
#endif
    /* return the final output after removing invalid features
     * the matching features between LT1 and LT2 are fLT1Re and fLT2
    */
    featurePointsLT1 = fLT1ReOffset;
    return flT2Offset;
}

/* triangulation; convert 2d feature points to 3d world points
*/
std::vector<cv::Point3f> VOClass::get3DPoints(std::vector<cv::Point2f> featurePoints, 
                                              cv::Mat disparityMap){
    std::vector<cv::Point3f> points3D;
    int numFeatures = featurePoints.size();
    cv::Mat point2D = cv::Mat::zeros(4, 1, CV_32F);
    cv::Mat point3DHC = cv::Mat::zeros(4, 1, CV_32F);

    /* Q (4x4) * (u, v, disp, 1) = 3d point (4x1)
    */
    for(int i = 0; i < numFeatures; i++){
        /* construct the 4x1 vector
        */
        point2D.at<float>(0, 0) = featurePoints[i].x;
        point2D.at<float>(1, 0) = featurePoints[i].y;
        point2D.at<float>(2, 0) = disparityMap.at<float>(featurePoints[i].x, featurePoints[i].y);
        /* discard points with <=0 disparity
        */
        if(point2D.at<float>(2, 0) <= 0)
            continue;

        point2D.at<float>(3, 0) = 1;
        /* multiply to get 4x1 vector
        */
        point3DHC = qMat * point2D;
        /* divide by scale factor
        */
        cv::Point3f point3DNonHC;
        point3DNonHC.x = point3DHC.at<float>(0, 0)/point3DHC.at<float>(3, 0);
        point3DNonHC.y = point3DHC.at<float>(1, 0)/point3DHC.at<float>(3, 0);
        point3DNonHC.z = point3DHC.at<float>(2, 0)/point3DHC.at<float>(3, 0);

        points3D.push_back(point3DNonHC);
#if 0
        Logger.addLog(Logger.levels[TEST], "3d point calculation");
        Logger.addLog(Logger.levels[TEST], "qMat (4x4)");
        for(int r = 0; r < 4; r++){
            Logger.addLog(Logger.levels[TEST], qMat.at<float>(r, 0), 
                                               qMat.at<float>(r, 1), 
                                               qMat.at<float>(r, 2), 
                                               qMat.at<float>(r, 3));
        }
        Logger.addLog(Logger.levels[TEST], "[u,v,disp,1]");
        for(int r = 0; r < 4; r++){
            Logger.addLog(Logger.levels[TEST], point2D.at<float>(r, 0)); 
        }
        Logger.addLog(Logger.levels[TEST], "Homogeneous 3D point");
        for(int r = 0; r < 4; r++){
            Logger.addLog(Logger.levels[TEST], point3DHC.at<float>(r, 0)); 
        }
        Logger.addLog(Logger.levels[TEST], "Non homogeneous 3D point");
        Logger.addLog(Logger.levels[TEST], point3DNonHC.x, point3DNonHC.y, point3DNonHC.z);
#endif
    }
#if 0
    cv::Mat colors;
    cv::cvtColor(imgLT1, colors, cv::COLOR_GRAY2RGB);
    writeToPLY(points3D, colors);
#endif
    Logger.addLog(Logger.levels[INFO], "Computed 3D points", points3D.size());
    return points3D;    
}



