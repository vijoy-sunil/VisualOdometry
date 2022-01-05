/* Implementation of the logger class.
 * The header has the set log levels and the use cases for 
 * these levels.
 * 
 * You have to maintain the parameter style while passing them 
 * in the logging function in order to maintain uniformity.
*/

#include "../Include/Logger.h"
#include "../Include/Constants.h"
/* instantiate this module, there is only one instance of this
 * module
*/
LoggerClass Logger(dumpFilePath);

LoggerClass::LoggerClass(const std::string filePath){
    /* open the log file, assert if unable to open
    */
    logFile.open(filePath);
    if(!logFile.is_open())
        assert(false);
}

LoggerClass::~LoggerClass(void){
    logFile.close();
}