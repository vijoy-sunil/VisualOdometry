#ifndef LOGGER_H
#define LOGGER_H

/* Include this header to use the logger functions
*/

#include <cassert>
/* File I/O operations
 * fstream header file
*/
#include <fstream>
#include <string>

/* Log levels from low to high
 * DEBUG - The DEBUG log level should be used for 
 * information that may be needed for diagnosing 
 * issues and troubleshooting
 * [use cases - dump functions]
 * 
 * INFO – the standard log level indicating that 
 * something happened, the application entered a 
 * certain state, etc. The information logged using 
 * the INFO log level should be purely informative 
 * and not looking into them on a regular basis 
 * shouldn’t result in missing any important information.
 * [use cases - upon accomplishing something]
 * 
 * WARN - The WARN level should be used in situations 
 * that are unexpected, but the code can continue the work.
 * [use cases - when you return false from a function]
 * 
 * ERROR – the log level that should be used when the 
 * application hits an issue preventing one or more 
 * functionalities from properly functioning. 
 * [use cases - when you assert]
 * 
 * TEST - this log level indicates that the log entry 
 * is from a test run.
 * [use cases - in test files and test results]
*/
typedef enum{
    DEBUG, 
    INFO, 
    WARNING, 
    ERROR,
    TEST
}LoggerLevel;

/* number of log levels, this is used to
 * init the logger level string
*/
const int numLevels = 5;

class LoggerClass{
    private:
        /* This data type represents the output file stream 
         * and is used to create files and to write information 
         * to files.
        */
        std::ofstream logFile;
        /* To handle base case of the recursive 
        * variadic function Template
        * This will be called after the last parameter.
        */
        void addLog(void){
            logFile<<std::endl;
        }

    public:
        /* correlating level enums with level strings
        */
        std::string levels[numLevels] = {
            "[ DEB ]", "[ INF ]", "[ WRN ]", "[ ERR ]", "[ TST ]"};

        LoggerClass(const std::string filePath);
        ~LoggerClass(void);
        /* recursive variadic function to pass in 
         * variable number and types of arguments.
         *
         * ellipsis (...) operator to the left of 
         * the parameter name declares a parameter 
         * pack, allowing you to declare zero or more 
         * parameters (of different types).
         * 
         * Maintain this format when passing in
         * parameters:
         * [LEVEL] [FUNCTION NAME] [PARM1] [PARAM2] ...
         * 
         * NOTE: Turns out templates may only be implemented 
         * in the header file.
        */
        template<typename First, typename... Args>
        void addLog(First first, Args... args){
            if(logFile.is_open()){
                logFile<<first<<" ";
                addLog(args...);
            }
        }
};

/* instantiate this module once in the .cpp file.
 * Include this header file to use this module.
*/
extern LoggerClass Logger;
#endif /* LOGGER_H
*/