#include <string>
#include <iostream>
#include <cmath>

#include "seqGeneralHoughTransform.h"
#include "utils.h"

const int THRESHOLD = 50;
const float PI = 3.14159265;
const int nRotationSlices = 72;
const float deltaRotationAngle = 360 / nRotationSlices;
const float MAXSCALE = 2.0f;
const float deltaScaleRatio = 0.5f;
const int nScaleSlices = MAXSCALE / deltaScaleRatio;

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
    // writeGrayPPMImage(grayTpl, "gray.ppm");

    // apply sobel_x -> gradient in x
    std::vector<std::vector<int>> sobelX = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    GrayImage* gradientX = new GrayImage;
    convolve(sobelX, grayTpl, gradientX);
    // writeGrayPPMImage(gradientX, "gx.ppm");

    // apply soble_y -> gradient in y
    std::vector<std::vector<int>> sobelY = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
    GrayImage* gradientY = new GrayImage;
    convolve(sobelY, grayTpl, gradientY);
    // writeGrayPPMImage(gradientY, "gy.ppm");

    // grascale magnitude = sqrt(square(x) + square(y))
    GrayImage* mag = new GrayImage;
    magnitude(gradientX, gradientY, mag);
    // writeGrayPPMImage(mag, "mag.ppm");

    // grascale orientation = (np.degrees(np.arctan2(image_y,image_x))+360)%360
    // orientation in [0, 360)
    GrayImage* orient = new GrayImage;
    orientation(gradientX, gradientY, orient);
    // writeGrayPPMImage(orient, "orient.ppm");

    // apply a threshold to get a binary image (255 is edge)
    GrayImage* magThreshold = new GrayImage;
    threshold(mag, magThreshold, THRESHOLD);
    // writeGrayPPMImage(magThreshold, "threshold.ppm");

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
    // convert source to gray image
    GrayImage* graySrc = new GrayImage;
    convertToGray(src, graySrc);

    // apply sobel_x -> grayscale image x
    std::vector<std::vector<int>> sobelX = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    GrayImage* gradientX = new GrayImage;
    convolve(sobelX, graySrc, gradientX);

    // apply soble_y -> grayscale image y
    std::vector<std::vector<int>> sobelY = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
    GrayImage* gradientY = new GrayImage;
    convolve(sobelY, graySrc, gradientY);

    // grascale magnitude = sqrt(square(x) + square(y))
    GrayImage* mag = new GrayImage;
    magnitude(gradientX, gradientY, mag);

    // grascale gradient = (np.degrees(np.arctan2(image_y,image_x))+360)%360
    GrayImage* orient = new GrayImage;
    orientation(gradientX, gradientY, orient);

    // apply a threshold to get a binary image (255 is edge)
    GrayImage* magThreshold = new GrayImage;
    threshold(mag, magThreshold, THRESHOLD);
    // -------Reuse from processTemplate ends-------

    // GrayImage* phiDirection = new GrayImage;
    // phiDirection(orient, magThreshold, phiDirection);

    // initialize accumulator array (4D: Scale x Rotation x Width x Height)
    int width = magThreshold->width;
    int height = magThreshold->height;
    std::vector<std::vector<std::vector<std::vector<int>>>> accumulator(nScaleSlices, std::vector<std::vector<std::vector<int>>>(nRotationSlices, std::vector<std::vector<int>>(width, std::vector<int>(height, 0))));

    // Each edge pixel vote
    printf("------Start calculating accumulator-------\n");
    int totalEgdes = 0;
    for (int j = 0 ; j < height; j++) {
        for (int i = 0 ; i < width; i++) {
            if (magThreshold->data[j * width + i] > 254) {
                totalEgdes++;
                // calculate edge gradient
                float phi = orient->data[j * width + i]; // gradient direction in [0,360)
                int iSlice = static_cast<int>(phi / deltaRotationAngle);
                std::vector<rEntry> entries = rTable[iSlice];
                for (int k = 0 ; k < entries.size(); k++){
                    float r = sqrt(entries[k].x*entries[k].x + entries[k].y*entries[k].y);
                    float alpha = entries[k].alpha;
                    for (int is = 1; is <= nScaleSlices; is++){
                        float s = is*deltaScaleRatio;
                        for (int itheta = 0; itheta < nRotationSlices; itheta++){
                            float theta = itheta*deltaRotationAngle/(2*PI);
                            int xc = i + round(r*s*cos(alpha+theta));
                            int yc = j + round(r*s*sin(alpha+theta));
                            if (xc>=0 && xc<width && yc>=0 && yc<height){
                                accumulator[is-1][itheta][xc][yc]++;
                            }
                        }
                    }
                }
            }
        }
    }
    // printf("total edges: %d\n", totalEgdes);
    printf("------End calculating accumulator-------\n");

    // find local maxima
    for (int is = 1; is <= nScaleSlices; is++){
        for (int itheta = 0; itheta < nRotationSlices; itheta++){
            int maximaThres = round(totalEgdes*is*deltaScaleRatio*0.7);
            //printf("s: %f, theta: %f, maxima threshold: %d\n", is*deltaScaleRatio, itheta*deltaRotationAngle, maximaThres);
            for (int i = 0; i < width ; i++){
                for (int j = 0; j < height; j++){
                    if (localMaxima(accumulator[is-1][itheta], i, j, maximaThres)){
                        hitPoints.push_back((struct Point){i, j});
                        printf("hit points: %d %d hits: %d\n", i, j, accumulator[is-1][itheta][i][j]);
                    }
                }
            }
        }
    }

    // memory deallocation
    delete graySrc;
    delete gradientX;
    delete gradientY;
    delete mag;
    delete orient;
    delete magThreshold;
}

bool SeqGeneralHoughTransform::localMaxima(std::vector<std::vector<int>> accum, int i, int j, int maximaThres){
    if (accum[i][j] < maximaThres) return false;
    int boundRight = (i+1 < accum.size())? i+1 : accum.size()-1;
    int boundLeft = (i-1 >= 0)? i-1 : 0;
    int boundTop = (j+1 < accum.size())? j+1 : accum[0].size()-1;
    int boundBottom = (j-1 >= 0)? j-1 : 0;
    for (int ii = boundLeft; ii <= boundRight ; ii++){
        for (int jj = boundBottom; jj <= boundTop; jj++){
            if (ii==i && jj==j) continue;
            if (accum[ii][jj] >= accum[i][j]) return false;
        }
    }
    return true;
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

void SeqGeneralHoughTransform::convolve(std::vector<std::vector<int>> filter, const GrayImage* source, GrayImage* result) {
    if (filter.size() != 3 || filter[0].size() != 3) {
        std::cerr << "ERROR: convolve() only supports 3x3 filter.\n";
        return;
    }
    result->setGrayImage(source->width, source->height);
    for (int j = 0; j < source->height; j++) {
        for (int i = 0; i < source->width; i++) {
            float tmp = 0.f;
            for (int jj = -1; jj <= 1; jj++) {
                for (int ii = -1; ii <= 1; ii++) {
                    int row = j + jj;
                    int col = i + ii;
                    if (row < 0 || row >= source->height || col < 0 || col >= source->width) {
                        // out of image bound, do nothing
                    } else {
                        int idx = row * source->width + col;
                        tmp += source->data[idx] * filter[jj + 1][ii + 1];
                    }
                }
            }
            result->data[j * source->width + i] = tmp;
        }
    }
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

// void SeqGeneralHoughTransform::phiDirection(const GrayImage* orientation, const GrayImage* magThreshold, GrayImage* result) {
//     result->setGrayImage(orientation->width, orientation->height);
//     for (int i = 0; i < orientation->width * orientation->height; i++) {
//         if (magThreshold->data[i] > 0) result->data[i] = orientation->data[i];
//         else result->data[i] = 0;
//     }
// }

void SeqGeneralHoughTransform::createRTable(const GrayImage* orientation, const GrayImage* magThreshold) {
    rTable.clear();
    rTable.resize(nRotationSlices);

    centerX = orientation->width / 2;
    centerY = orientation->height / 2;

    for (int j = 0 ; j < orientation->height; j++) {
        for (int i = 0 ; i < orientation->width; i++) {
            if (magThreshold->data[j * orientation->width + i] > 254) {
                float phi = orientation->data[j * orientation->width + i]; // gradient direction in [0,360)
                int iSlice = static_cast<int>(phi / deltaRotationAngle);
                rTable[iSlice].push_back((struct rEntry){centerX - i, centerY - j, static_cast<float>(atan2(centerY - j, centerX - i))});
            }
        }
    }
}
