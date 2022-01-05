#include "../Include/VOClass.h"
#include "../Include/Constants.h"
#include <iostream>

int main(void){
    VOClass VO;
    /* read from input files
    */
    VO.getProjectionMatrices(calibrationFilePath);
    VO.getGroundTruthPath(groundTruthFilePath);

    /* main loop
    */
    for(int i = 1; i < numFrames; i++){
        /* read stereo image at t and t+1
        */
        VO.readStereoImagesT1T2(i);
        /* compute disparity map
        */
        cv::Mat disparityMapT1 = VO.computeDisparity(VO.imgLT1, VO.imgRT1);
        cv::Mat disparityMapT2 = VO.computeDisparity(VO.imgLT2, VO.imgRT2);
        /* detect features in imgLT1
         * NOTE: point for optimization \ :: /
        */
        std::vector<cv::Point2f> featurePointsT1 = VO.getFeaturesFAST(VO.imgLT1);
        /* match feature points imgLT1 -> imgLT2
        */
        std::vector<cv::Point2f> featurePointsT2 = VO.matchFeatureKLT(featurePointsT1);
        /* triangulate 3d points
        */
        
    }
    return 0;
}