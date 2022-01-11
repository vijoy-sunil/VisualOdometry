#include "../Include/VOClass.h"
#include "../Include/Logger.h"
#include "../Include/Constants.h"

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
    countValidMatches(status0), countValidMatches(status1), 
    countValidMatches(status2), countValidMatches(status3));   

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
    countValidMatches(status0), countValidMatches(status1), 
    countValidMatches(status2), countValidMatches(status3));
 
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
