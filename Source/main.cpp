#include "../Include/VOClass.h"
#include "../Include/Constants.h"
#include "../Include/Logger.h"
#include <iostream>

int main(void){
    int numFrames = 0;
    VOClass VO;
    /* read from input files
    */
    VO.getProjectionMatrices(calibrationFilePath);
    VO.getGroundTruthPath(groundTruthFilePath, numFrames);
    /* Instead of running through the entire range of frames, run the
     * application for only a limited number of frames
    */
#if LIMITED_FRAMES_TEST_MODE
    numFrames = 2;
#endif
    assert(numFrames != 1);
    /* output trajectory
    */
    std::vector<cv::Mat> estimatedTrajectory;
    /* first element in trajectory has to be (0, 0, 0)
    */
    estimatedTrajectory.push_back(cv::Mat::zeros(3, 1, CV_64F));
    /* main loop
    */
    for(int i = 0; i < numFrames-1; i++){
        /* read stereo image at t and t+1
        */
        VO.readStereoImagesT1T2(i);
        /* compute disparity and depth map
        */
        cv::Mat disparityMapT1 = VO.computeDisparity(VO.imgLT1, VO.imgRT1);
        cv::Mat depthMapT1 = VO.computeDepthMap(disparityMapT1);
        /* detect features in imgLT1
        */
        std::vector<cv::Point2f> featurePointsT1 = VO.getFeaturesFAST(VO.imgLT1);
        /* match feature points imgLT1 -> imgLT2
        */
        std::vector<cv::Point2f> featurePointsT2 = VO.matchFeatureKLT(featurePointsT1);
        /* estimate motion, use depthMapT1 to convert featurePointsT1 to 
         * 3D points in camera frame in order to estimate motion
        */
        estimatedTrajectory.push_back(VO.estimateMotion(featurePointsT1, featurePointsT2, depthMapT1));
    }
    /* compute error between estimated trajectory and ground truth
    */
    float error = VO.computeErrorInPoseEstimation(estimatedTrajectory);
    Logger.addLog(Logger.levels[INFO], "Measured error", error);
    std::cout<<"Measured error: "<<error<<std::endl;

#if SHOW_GROUND_TRUTH_AND_ESTIMATED_TRAJECTORY
    /* plot trajectory
    */
    VO.testShowTrajectoryPair(estimatedTrajectory);
#endif
    return 0;
}