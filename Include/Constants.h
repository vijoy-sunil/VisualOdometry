#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>

/* Macros
*/
#define LIMITED_FRAMES_TEST_MODE                    1
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

/* choose the set of images to use in the KITTI dataset
*/
const std::string sequenceID = "00";
/* file names
*/
const std::string calibrationFile= "calib.txt";
const std::string dumpFile = "log.txt";
const std::string plyFile = "pointCloud.ply";
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
#endif /* CONSTANTS_H
*/
