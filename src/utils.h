#pragma once
#include "image.h"
#include <string>

struct rEntry {
    float x;
    float y;
    float alpha; // arctan2 is expensive, only do it once
};

bool readPPMImage(std::string filename, Image* result);

void writePPMImage(const Image* image, std::string filename);

void convertToGray(const Image* image, GrayImage& result);