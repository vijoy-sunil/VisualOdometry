#include "../Include/VOClass.h"
#include "../Include/Constants.h"

int main(void){
    VOClass VO;
    /* read from input files
    */
    VO.getProjectionMatrices(calibrationFilePath);
    VO.getGroundTruthPath(groundTruthFilePath);

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
        VO.estimateMotion(featurePointsT1, featurePointsT2, depthMapT1);
    }
    return 0;
}