#pragma once
#include "image.h"
#include <stdlib.h>

class GeneralHoughTransform {
public: 
    virtual ~GeneralHoughTransform(){};

    virtual void processTemplate() = 0;

    virtual void accumulateSource() = 0;

    virtual bool loadTemplate(std::string filename) = 0;

    virtual bool loadSource(std::string filename) = 0;

private:
    // Apply a kxk filter
    virtual void convolve(int **filter, size_t k, const GrayImage* source, GrayImage& result) = 0;
};
