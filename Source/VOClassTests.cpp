#include "../Include/VOClass.h"
#include "../Include/Logger.h"

/* this displays the stereo pair side by side
*/
void VOClass::testShowStereoImage(cv::Mat imgLeft, cv::Mat imgRight, int frameNumber){
    Logger.addLog(Logger.levels[TEST], "Show stereo pair images", frameNumber);

    cv::Mat imgPair;
    cv::vconcat(imgLeft, imgRight, imgPair);

    /* Mat, text, point, font, font scale, color, thickness
    */
    std::string text = "FRAME: " + std::to_string(frameNumber);
    cv::putText(imgPair, text, cv::Point(20, 40), cv::FONT_HERSHEY_DUPLEX, 
                1, cv::Scalar(255, 255, 255), 1
    );
    imshow("Stereo Pair", imgPair);
    cv::waitKey(0);
}

/* stereo pair, disparity map
*/
void VOClass::testShowDisparityImage(cv::Mat imgLeft, cv::Mat imgRight, cv::Mat disparityMap){
    Logger.addLog(Logger.levels[TEST], "Show disparity and stereo pair images");

    cv::Mat imgPair, imgTuple;
    cv::vconcat(imgLeft, imgRight, imgPair);

    /* imgPair is of type CV_8U, to concat, both of them need to be of the 
     * same type
    */
    cv::Mat disparityMapTypeChanged;
    disparityMap.convertTo(disparityMapTypeChanged, CV_8U);

    cv::vconcat(imgPair, disparityMapTypeChanged, imgTuple);
    imshow("Disparity Map", imgTuple);
    cv::waitKey(0);
}

/* disparity, depth map
*/
void VOClass::testShowDepthImage(cv::Mat disparityMap, cv::Mat depthMap){
    Logger.addLog(Logger.levels[TEST], "Show disparity and depth images");

    cv::Mat imgPair;
    /* convert to cv_8U to display properly
    */
    cv::Mat disparityMapTypeChanged, depthMapTypeChanged;
    disparityMap.convertTo(disparityMapTypeChanged, CV_8U);
    depthMap.convertTo(depthMapTypeChanged, CV_8U);

    cv::vconcat(disparityMapTypeChanged, depthMapTypeChanged, imgPair);
    imshow("Depth Map", imgPair);
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

/* display result of circular matching
*/
void VOClass::testShowCirculatMatchingPair(cv::Mat img, 
                                   std::vector<cv::Point2f> featurePointsCurrent, 
                                   std::vector<cv::Point2f> featurePointsNext, 
                                   std::vector<unsigned char> status){
    /* even if they are of the same size, not all of them would be
     * matched, we need to look at the status vector to confirm match
    */
    assert(featurePointsCurrent.size() == featurePointsNext.size());
    Logger.addLog(Logger.levels[TEST], "Show circular matching result");
    /* img is a single channel image, we will convert it to 3
     * to color our feature points and lines
    */
    cv::Mat imgChannelChanged;
    cv::cvtColor(img, imgChannelChanged, cv::COLOR_GRAY2RGB); 
    /* line connecting current feature to prev feature point to show
     * optical flow
    */    
    for(int i = 0; i < featurePointsCurrent.size(); i++){
        if(status[i] == 1){
            cv::line(imgChannelChanged, featurePointsCurrent[i], featurePointsNext[i], 
            cv::Scalar(0, 255, 0), 2);
        }
    }
    imshow("Tracked features", imgChannelChanged);
    cv::waitKey(0);
}

/* display full circular matching
*/
void VOClass::testShowCirculatMatchingFull(std::vector<cv::Point2f> fLT1, 
                                           std::vector<cv::Point2f> fRT1, 
                                           std::vector<cv::Point2f> fRT2, 
                                           std::vector<cv::Point2f> fLT2,
                                           std::vector<cv::Point2f> fLT1Re){

    int n = fLT1.size();
    Logger.addLog(Logger.levels[TEST], "Show full circular matching result");
    int idx = 4000;

    assert(idx < n);
    /* change all 4 images to 3 channels
    */
    cv::Mat img1, img2, img3, img4;
    cv::cvtColor(imgLT1, img1, cv::COLOR_GRAY2RGB); 
    cv::cvtColor(imgRT1, img2, cv::COLOR_GRAY2RGB); 
    cv::cvtColor(imgRT2, img3, cv::COLOR_GRAY2RGB); 
    cv::cvtColor(imgLT2, img4, cv::COLOR_GRAY2RGB);
    /* annotate frame
    */
    std::string text[4] = {"imgLT1", "imgRT1", "imgRT2", "imgLT2"};
    cv::putText(img1, text[0], cv::Point(20, 40), cv::FONT_HERSHEY_DUPLEX, 
                1, cv::Scalar(255, 255, 255), 1);
    cv::putText(img2, text[1], cv::Point(20, 40), cv::FONT_HERSHEY_DUPLEX, 
                1, cv::Scalar(255, 255, 255), 1);
    cv::putText(img3, text[2], cv::Point(20, 40), cv::FONT_HERSHEY_DUPLEX, 
                1, cv::Scalar(255, 255, 255), 1);
    cv::putText(img4, text[3], cv::Point(20, 40), cv::FONT_HERSHEY_DUPLEX, 
                1, cv::Scalar(255, 255, 255), 1);

    /* mark feature in frame before connecting lines
    */
    cv::circle(img1, fLT1[idx], 5, CV_RGB(0, 0, 255), 2);
    cv::circle(img1, fLT1Re[idx], 10, CV_RGB(255, 0, 0), 2);
    cv::circle(img2, fRT1[idx], 5, CV_RGB(0, 0, 255), 2);
    cv::circle(img3, fRT2[idx], 5, CV_RGB(0, 0, 255), 2);
    cv::circle(img4, fLT2[idx], 5, CV_RGB(0, 0, 255), 2);

    cv::Mat imgPair1, imgPair2;
    cv::hconcat(img1, img2, imgPair1);
    /* connect f1 to f2
    */
    cv::line(imgPair1, fLT1[idx], cv::Point(frameW + fRT1[idx].x, fRT1[idx].y), 
    cv::Scalar(0, 255, 0), 2);
    /* connect f3 to f4
    */
    cv::hconcat(img4, img3, imgPair2);
    cv::line(imgPair2, fLT2[idx], cv::Point(frameW + fRT2[idx].x, fRT2[idx].y), 
    cv::Scalar(0, 255, 0), 2);    

    cv::Mat imgQuad;
    cv::vconcat(imgPair1, imgPair2, imgQuad);
    /* connect f2 to f3
    */
    cv::line(imgQuad, cv::Point(frameW + fRT1[idx].x, fRT1[idx].y), 
                      cv::Point(frameW + fRT2[idx].x, frameH + fRT2[idx].y), 
                      cv::Scalar(0, 255, 0), 2);
    /* connect f1 to f4
    */
    cv::line(imgQuad, cv::Point(fLT1Re[idx].x, fLT1Re[idx].y), 
                      cv::Point(fLT2[idx].x, frameH + fLT2[idx].y), 
                      cv::Scalar(0, 0, 255), 2);    

    imshow("Full Circular Matching", imgQuad);
    cv::waitKey(0);
}

