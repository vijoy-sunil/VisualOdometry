#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>

/* When set, the application reads pose output from previous run and
 * plots it agains ground truth
*/
#define READ_ESTIMATED_POSE_FILE                    1
/* Debug macros
*/
#define LIMITED_FRAMES_TEST_MODE                    0
#define SHOW_STEREO_IMAGE_PAIR                      0
#define SHOW_GROUND_TRUTH_TRAJECTORY                0
#define SHOW_DISPARITY_MAP                          0
#define WRITE_DEPTH_PLY_FILE                        0
#define SHOW_DEPTH_MAP                              0
#define SHOW_ALL_FAST_FEATURES                      0
#define SHOW_CIRCULAR_MATCHING_PAIR                 0
#define SHOW_CIRCULAR_MATCHING_PAIR_BOUNDS_FILTER   0
#define SHOW_CIRCULAR_MATCHING_QUAD                 0
#define SHOW_ALL_FAST_FEATURES_STABLE               0
#define POSE_ESTIMATION_RANSAC                      1
#define SHOW_GROUND_TRUTH_AND_ESTIMATED_TRAJECTORY  1
/* This is done so that it doesn't clear contents of saved output pose
 * file
*/
#if READ_ESTIMATED_POSE_FILE
    #define WRITE_ESTIMATED_POSE_FILE               0
#else
    #define WRITE_ESTIMATED_POSE_FILE               1   
#endif

/* choose the set of images to use in the KITTI dataset; 00 to 10
 * sets of data
*/
const std::string sequenceID = "00";
/* limited frame mode
*/
const int limitedFramesCount = 2;
/* file names
*/
const std::string calibrationFile= "calib.txt";
const std::string dumpFile = "log.txt";
const std::string plyFile = "pointCloud.ply";
const std::string estiamtedPoseFile = "outputPoses.txt";
/* paths
*/
const std::string datasetPath = "../../Data/sequences/";
const std::string posePath = "../../Data/poses/";
const std::string leftImagesDir = "image_0/";
const std::string rightImagesDir = "image_1/";
const std::string dataPath = "../Log/";

const std::string calibrationFilePath = datasetPath +  sequenceID + "/" + calibrationFile;
const std::string groundTruthFilePath = posePath + sequenceID + ".txt";
const std::string leftImagesPath = datasetPath + sequenceID + "/" + leftImagesDir;
const std::string rightImagesPath = datasetPath + sequenceID + "/" + rightImagesDir;
const std::string dumpFilePath = dataPath + dumpFile;
const std::string plyFilePath = dataPath + plyFile;
const std::string estiamtedPoseFilePath = dataPath + estiamtedPoseFile;
#endif /* CONSTANTS_H
*/
