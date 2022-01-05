#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>

/* choose the set of images to use in the KITTI dataset
*/
const std::string sequenceID = "00";
const int numFrames = 2;
/* file names
*/
const std::string calibrationFile= "calib.txt";
const std::string dumpFile = "log.txt";
/* paths
*/
const std::string datasetPath = "../Data/sequences/";
const std::string posePath = "../Data/poses/";
const std::string leftImagesDir = "image_0/";
const std::string rightImagesDir = "image_1/";
const std::string dataPath = "../Data/";

const std::string calibrationFilePath = datasetPath +  sequenceID + "/" + calibrationFile;
const std::string groundTruthFilePath = posePath + sequenceID + ".txt";
const std::string leftImagesPath = datasetPath + sequenceID + "/" + leftImagesDir;
const std::string rightImagesPath = datasetPath + sequenceID + "/" + rightImagesDir;
const std::string dumpFilePath = dataPath + dumpFile;
#endif /* CONSTANTS_H
*/
