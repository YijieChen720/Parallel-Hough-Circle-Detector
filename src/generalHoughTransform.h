#pragma once
#include "image.h"

class GeneralHoughTransform {
public: 
    virtual ~GeneralHoughTransform() {};

    bool loadTemplate(std::string filename);

    bool loadSource(std::string filename);

private:
    bool loadImage(std::string filename);
    
    // Apply a kxk filter
    virtual void convolve(int **filter, size_t k, Image& result) = 0;
};
