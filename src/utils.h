#pragma once
#include "image.h"
#include <string>

struct rEntry {
    int x;
    int y;
    float alpha; // arctan2 is expensive, only do it once
};

struct Point{
    int x;
    int y;
};

bool readPPMImage(std::string filename, Image* result);

void writePPMImage(const Image* image, std::string filename);

void writeGrayPPMImage(const GrayImage* image, std::string filename);
