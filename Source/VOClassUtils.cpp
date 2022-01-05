#include "../Include/VOClass.h"
#include "../Include/Utils.h"

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

bool VOClass::isOutOfBounds(cv::Point2f featurePoint){
    if(featurePoint.x < 0 || featurePoint.x > frameW)
        return true;
    
    if(featurePoint.y < 0 || featurePoint.y > frameH)
        return true;

    return false;
}

void VOClass::markInvalidFeaturesBounds(std::vector<cv::Point2f> featurePoints, 
                                    std::vector<unsigned char>& status){
    int numFeatures = featurePoints.size();
    for(int i = 0; i < numFeatures; i++){
        if(isOutOfBounds(featurePoints[i]))
            status[i] = 0;
    }
}

int VOClass::validMatches(std::vector<unsigned char> status){
    int n = status.size();
    int numOnes = 0;
    for(int i = 0; i < n; i++){
        /* A feature point is only matched correctly if the status
         * is set
        */
        if(status[i] == 1)
            numOnes++;
    }
    return numOnes;
}

void VOClass::removeInvalidFeatures(std::vector<cv::Point2f>& featurePointsPrev, 
                                    std::vector<cv::Point2f>& featurePointsCurrent, 
                                    std::vector<unsigned char> status){
    /* create an empty feature vector, push valid ones into this and
     * finally copy this to the original
    */
    std::vector<cv::Point2f> validPointsPrev, validPointsCurrent;
    int numFeatures = featurePointsPrev.size();
    for(int i = 0; i < numFeatures; i++){
        if(status[i] == 1){
            validPointsPrev.push_back(featurePointsPrev[i]);
            validPointsCurrent.push_back(featurePointsCurrent[i]);
        }
    }
    featurePointsPrev = validPointsPrev;
    featurePointsCurrent = validPointsCurrent;
}
