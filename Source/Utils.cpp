#include "../Include/Utils.h"
#include <iostream>

std::vector<std::string> tokenize(std::string str, std::string del){
    std::vector<std::string> sub;
    int start = 0;
    int end = str.find(del);
    while (end != -1) {
        
        sub.push_back(str.substr(start, end - start));
        start = end + del.size();
        end = str.find(del, start);
    }
    sub.push_back(str.substr(start, end - start));
    return sub;
}

std::string formatStringWidth(int num, int width){
    /* start with a string full of 0s with the set width
    */
    std::string result(width--, '0');
    for (int val = num; width >= 0 && val != 0; --width, val/=10){
        /* extract last digit and convert to char, insert to
         * result string
        */
       result[width] = '0'+ (val % 10);
    }
    return result;
}

void showProgressBar(float progress){
    const int barWidth = 70;
    std::cout << "[";

    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; i++){
        if (i < pos) 
            std::cout << "=";
        else if (i == pos) 
            std::cout << ">";
        else 
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();  
    /* this is done so that the next cout gets printed on the next line
    */
    if(progress == 1.0)
        std::cout<<std::endl;
}