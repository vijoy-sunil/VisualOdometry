#ifndef VOCLASS_H
#define VOCLASS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class VOClass{
    private:
        /* input frame dimensions
        */
        int frameW;
        int frameH;
        /* 3x4 projection matrix of camera1 (left) and camera2
         * (right); we call the cameras CL and CR
         * This consists of both intrinsic and extrinsic params, 
         * and is used to transform 3D world coordinates to 2d image 
         * coordinates, and we get this after the calibration process.
        */
        cv::Mat projectionCL = cv::Mat::zeros(3, 4, CV_32F);
        cv::Mat projectionCR = cv::Mat::zeros(3, 4, CV_32F);
        /* read from calib file and store into matrix; this fn
         * is called by getProjectionMatrices()
        */
        void constructProjectionMatrix(std::string line, cv::Mat& projectionMat);
        /* extrinsic matrix used to find camera pose as ground truth
        */
        cv::Mat extrinsicMat = cv::Mat::zeros(4, 4, CV_32F);
        /* vector to hold ground truth poses
        */
        std::vector<float> groundX, groundY, groundZ;
        /* read from poses.txt and store it into matrix
        */
        void constructExtrinsicMatrix(std::string line);
        /* Q matrix
        */
        cv::Mat qMat = cv::Mat::zeros(4, 4, CV_32F);
        /* read from projection matrix
        */
        void constructQMatrix(void);
        /* extract R and T from extrinsic matrix
        */
        void extractRT(cv::Mat& R, cv::Mat& T);
        /* check if a feature is out of bounds
        */
        bool isOutOfBounds(cv::Point2f featurePoint);
        /* update status vector for out of bounds points
        */
        void markInvalidFeaturesBounds(std::vector<cv::Point2f> featurePoints, 
        std::vector<unsigned char>& status);
        /* get number of valid matches (count of one's) in status vector
        */
        int validMatches(std::vector<unsigned char> status);
        /* remove invalid features based on status vector
        */
        void removeInvalidFeatures(std::vector<cv::Point2f>& featurePointsPrev, 
                                   std::vector<cv::Point2f>& featurePointsCurrent, 
                                   std::vector<unsigned char> status);

    public:
        /* we need to hold 4 images at a time; 2x at time t and 
         * 2x at time (t+1)
        */
        cv::Mat imgLT1, imgRT1;
        cv::Mat imgLT2, imgRT2;

        VOClass(void);
        ~VOClass(void);
        /* read images from directory, manipulate file name based
         * on frame number
        */
        bool readStereoImagesT1T2(int frameNumber);
        /* construct projection matrices for both cameras from the
         * calibration file
        */
        bool getProjectionMatrices(const std::string calibrationFile);
        /* 4x4 Q matrix
        */
        void getQMatrix(void);
        /* get ground truth output poses, so that we can compare our
         * estimate with it at the end
        */
        bool getGroundTruthPath(const std::string groundTruthFile);
        /* compute disparity
        */
        cv::Mat computeDisparity(cv::Mat leftImg, cv::Mat rightImg);
        /* feature detection
        */
        std::vector<cv::Point2f> getFeaturesFAST(cv::Mat img);
        /* feature matching
        */
        std::vector<cv::Point2f> matchFeatureKLT(std::vector<cv::Point2f> &featurePointsLT1);
        /* test fns
        */
        void testShowStereoImage(cv::Mat imgLeft, cv::Mat imgRight, int frameNumber);
        void testShowDisparityImage(cv::Mat imgLeft, cv::Mat imgRight, cv::Mat disparityMap);
        void testShowGroundTruthTrajectory(void);
        void testShowDetectedFeatures(cv::Mat img, std::vector<cv::Point2f> featurePoints);
        void testShowCirculatMatchingPair(cv::Mat img, 
                                          std::vector<cv::Point2f> featurePointsCurrent, 
                                          std::vector<cv::Point2f> featurePointsNext, 
                                          std::vector<unsigned char> status);
        void testShowCirculatMatchingFull(std::vector<cv::Point2f> fLT1, 
                                          std::vector<cv::Point2f> fRT1, 
                                          std::vector<cv::Point2f> fRT2, 
                                          std::vector<cv::Point2f> fLT2,
                                          std::vector<cv::Point2f> fLT1Re);
};

#endif /* VOCLASS_H
*/
