#include <string>
#include <iostream>

#include "seqGeneralHoughTransform.h"
#include "utils.h"

SeqGeneralHoughTransform::SeqGeneralHoughTransform() {
    tpl = new Image;
    src = new Image;
}

SeqGeneralHoughTransform::~SeqGeneralHoughTransform() {
    if (tpl) delete tpl;
    if (src) delete src;
}

void SeqGeneralHoughTransform::processTemplate() {
    // convert template to gray image

    // apply sobel_x -> grayscale image x

    // apply soble_y -> grayscale image y

    // grascale magnitude = sqrt(square(x) + square(y))

    // grascale gradient = (np.degrees(np.arctan2(image_y,image_x))+360)%360

    // apply a threshold to get a binary image (255 is edge)

    // create R table using gradient (to get Phi) and image center
}

void SeqGeneralHoughTransform::accumulateSource() {
    // -------Reuse from processTemplate-------
    // convert template to gray image

    // apply sobel_x -> grayscale image x

    // apply soble_y -> grayscale image y

    // grascale magnitude = sqrt(square(x) + square(y))

    // grascale gradient = (np.degrees(np.arctan2(image_y,image_x))+360)%360

    // apply a threshold to get a binary image (255 is edge)

    // -------Reuse from processTemplate ends-------

    // initialize accumulator array (4D: Width x Height x Scale x Rotation)

    // Each edge pixel vote

    // Find all local maxima
}

bool SeqGeneralHoughTransform::loadTemplate(std::string filename) {
    if (!readPPMImage(filename, tpl)) return false;
    return true;
}

bool SeqGeneralHoughTransform::loadSource(std::string filename) {
    if (!readPPMImage(filename, src)) return false;
    return true;
}

void SeqGeneralHoughTransform::convolve(int **filter, size_t k, const GrayImage* source, GrayImage& result) {
    
}