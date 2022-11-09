#pragma once
#include "image.h"
#include <string>

bool readPPMImage(std::string filename, Image& result);

void writePPMImage(const Image* image, std::string filename);