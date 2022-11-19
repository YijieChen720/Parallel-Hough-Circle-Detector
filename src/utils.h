#pragma once
#include "image.h"
#include <string>

struct rEntry {
    float r;  // sqrt is expensive, only do it once
    float alpha; // arctan2 is expensive, only do it once
};

struct Point{
    int x;
    int y;
    int hits;
    float scale;
    float rotation;
};

bool readPPMImage(std::string filename, Image* result);

void writePPMImage(const Image* image, std::string filename);

void writeGrayPPMImage(const GrayImage* image, std::string filename);

bool keepPixel(const GrayImage* magnitude, int i, int j, int gradient);
