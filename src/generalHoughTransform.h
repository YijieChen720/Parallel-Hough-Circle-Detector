#pragma once
#include "image.h"
#include <stdlib.h>

class GeneralHoughTransform {
public: 
    virtual ~GeneralHoughTransform(){};

    virtual void setup() = 0;

    virtual void processTemplate() = 0;

    virtual void accumulateSource() = 0;

    virtual void saveOutput() = 0;

    virtual bool loadTemplate(std::string filename) = 0;

    virtual bool loadSource(std::string filename) = 0;

};
