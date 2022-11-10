#include <string>
#include <iostream>
#include <cmath>

#include "seqGeneralHoughTransform.h"
#include "utils.h"

const int THRESHOLD = 30;
const float PI = 3.14159265;

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
    GrayImage* grayTpl = new GrayImage;
    convertToGray(tpl, grayTpl);

    // apply sobel_x -> gradient in x
    std::vector<std::vector<int>> sobelX = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    GrayImage* gradientX = new GrayImage;
    convolve(sobelX, 3, grayTpl, gradientX);

    // apply soble_y -> gradient in y
    std::vector<std::vector<int>> sobelY = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
    GrayImage* gradientY = new GrayImage;
    convolve(sobelY, 3, grayTpl, gradientY);

    // grascale magnitude = sqrt(square(x) + square(y))
    GrayImage* mag = new GrayImage;
    magnitude(gradientX, gradientY, mag);

    // grascale orientation = (np.degrees(np.arctan2(image_y,image_x))+360)%360
    GrayImage* orient = new GrayImage;
    orientation(gradientX, gradientY, orient);

    // apply a threshold to get a binary image (255 is edge)
    GrayImage* magThreshold = new GrayImage;
    threshold(mag, magThreshold, THRESHOLD);

    // update R table using gradient (Phi) and image center
    // also update centerX and centerY
    createRTable(orient, magThreshold);
    
    // memory deallocation
    delete grayTpl;
    delete gradientX;
    delete gradientY;
    delete mag;
    delete orient;
    delete magThreshold;
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

void SeqGeneralHoughTransform::convertToGray(const Image* image, GrayImage* result) {
    result->setGrayImage(image->width, image->height);
    for (int i = 0; i < image->width * image->height; i++) {
        result->data[i] = (image->data[3 * i] + image->data[3 * i + 1] + image->data[3 * i + 2]) / 3.f;
    }
}

void SeqGeneralHoughTransform::convolve(std::vector<std::vector<int>> filter, size_t k, const GrayImage* source, GrayImage* result) {

}

void SeqGeneralHoughTransform::magnitude(const GrayImage* gradientX, const GrayImage* gradientY, GrayImage* result) {
    result->setGrayImage(gradientX->width, gradientX->height);
    for (int i = 0; i < gradientX->width * gradientX->height; i++) {
        result->data[i] = sqrt(gradientX->data[i] * gradientX->data[i] + gradientY->data[i] * gradientY->data[i]);
    }
}

void SeqGeneralHoughTransform::orientation(const GrayImage* gradientX, const GrayImage* gradientY, GrayImage* result) {
    result->setGrayImage(gradientX->width, gradientX->height);
    for (int i = 0; i < gradientX->width * gradientX->height; i++) {
        result->data[i] = fmod(atan2(gradientY->data[i], gradientX->data[i]) * 180 / PI + 360, 360);
    }
}

void SeqGeneralHoughTransform::threshold(const GrayImage* magnitude, GrayImage* result, int threshold) {
    result->setGrayImage(magnitude->width, magnitude->height);
    for (int i = 0; i < magnitude->width * magnitude->height; i++) {
        if (magnitude->data[i] > threshold) result->data[i] = 255;
        else result->data[i] = 0;
    }
}

void SeqGeneralHoughTransform::createRTable(const GrayImage* orientation, const GrayImage* magnitude) {

}