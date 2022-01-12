#include "../Include/VOClass.h"
#include "../Include/Utils.h"
#include "../Include/Logger.h"
#include "../Include/Constants.h"

void VOClass::constructProjectionMatrix(std::string line, cv::Mat& projectionMat){
    /* split line to words
    */
    std::vector<std::string> sub = tokenize(line);
    /* skip first word
    */
    int k = 1;
    for(int r = 0; r < 3; r++){
        for(int c = 0; c < 4; c++){
            projectionMat.at<double>(r, c) = std::stof(sub[k++]);
        }
    }
}

void VOClass::constructExtrinsicMatrix(std::string line){
    /* split line to words
    */
    std::vector<std::string> sub = tokenize(line);
    int k = 0;
    /* Each line contains 12 values, and the number 12 comes from flattening
     * a 3x4 transformation matrix of the left stereo camera with respect to 
     * the global coordinate frame. 
     * 
     * A 3x4 transfomration matrix contains a 3x3 rotation matrix horizontally 
     * stacked with a 3x1 translation vector in the form R|t
     * 
     * [Xworld, Yworld, Zworld] = [R|t] * [Xcamera, Ycamera, Zcamera]
     * The camera's coordinate system is where the Z axis points forward, the 
     * Y axis points downwards, and the X axis is horizontal
    */
    for(int r = 0; r < 3; r++){
        for(int c = 0; c < 4; c++){
            extrinsicMat.at<double>(r, c) = std::stof(sub[k++]);
        }
    }
    /* add last row since that is omitted in the text file
     * last row will be [0, 0, 0, 1]
    */
    extrinsicMat.at<double>(3, 3) = 1;
}

void VOClass::extractRT(cv::Mat& R, cv::Mat& T){
    /* extract 3x3 R
    */
    for(int r = 0; r < 3; r++){
        for(int c = 0; c < 3; c++){
            R.at<double>(r, c) = extrinsicMat.at<double>(r, c);
        }
    }
    /* extract 3x1 R
    */
    for(int r = 0; r < 3; r++)
        T.at<double>(r, 0) = extrinsicMat.at<double>(r, 3);
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

int VOClass::countValidMatches(std::vector<unsigned char> status){
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

void VOClass::writeToPLY(cv::Mat depthMap, cv::Mat colors, int depthThresh, int numVertices){
    /* write to file
    */
    std::ofstream plyFile;
    plyFile.open(plyFilePath);
    if(plyFile.is_open()){
        /* file header
        */
        plyFile<<"ply"<<std::endl;
        plyFile<<"format ascii 1.0"<<std::endl;
        plyFile<<"element vertex "<<numVertices<<std::endl;
        plyFile<<"property float x"<<std::endl;
        plyFile<<"property float y"<<std::endl;
        plyFile<<"property float z"<<std::endl;
        plyFile<<"property uchar red"<<std::endl;
        plyFile<<"property uchar green"<<std::endl;
        plyFile<<"property uchar blue"<<std::endl;
        plyFile<<"end_header"<<std::endl;
        /* file contents
            * x, y, z, r, g, b
        */
        for(int r = 0; r < depthMap.rows; r++){
            for(int c = 0; c < depthMap.cols; c++){
                /* vertices
                */
                float z = depthMap.at<double>(r, c);
                /* filer z values beyond the threshold
                */
                if(z > depthThresh)
                    continue;
                plyFile<<r<<" "<<c<<" "<<-z<<" ";
                /* color values
                */
                cv::Vec3b color = colors.at<cv::Vec3b>(r, c);
                plyFile<<(int)color.val[0]<<" "
                       <<(int)color.val[1]<<" "
                       <<(int)color.val[2]<<std::endl;
            }  
        }
        Logger.addLog(Logger.levels[INFO], ".ply file write complete");
        plyFile.close();
    }
    else{
        Logger.addLog(Logger.levels[ERROR], "Unable to open .ply file");
        assert(false);
    }
}

int* VOClass::computeHistogram(cv::Mat src, int maxVal){
    /* src is a single channel image
    */
    /* we set the hist array size to maxVal + 1 to include the maxVal
     * itself in hitogram
    */
    int *hist = (int*)calloc((maxVal + 1), sizeof(int));
    for(int r = 0; r < src.rows; r++){
        for(int c = 0; c < src.cols; c++){
            int val = src.at<double>(r, c);
            hist[val]++;
        }
    }
    /* log histogram 
    */
    Logger.addLog(Logger.levels[INFO], "Computed histogram", maxVal, hist[maxVal]);
#if 0
    for(int i = 0; i <= maxVal; i++){
        if(hist[i] != 0)
            Logger.addLog(Logger.levels[DEBUG], i, hist[i]);
    }
#endif
    return hist;
}