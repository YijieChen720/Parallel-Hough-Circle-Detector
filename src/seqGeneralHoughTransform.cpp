#include <string>
#include <iostream>
#include <cmath>

#include "seqGeneralHoughTransform.h"
#include "utils.h"

const int THRESHOLD = 50;
const float PI = 3.14159265;
const int nRotationSlices = 72;
const float deltaRotationAngle = 360 / nRotationSlices;
const float MAXSCALE = 1.2f;
const float MINSCALE = 0.8f;
const float deltaScaleRatio = 0.1f;
const int nScaleSlices = (MAXSCALE - MINSCALE) / deltaScaleRatio + 1;
const int blockSize = 10;
const float thresRatio = 0.9;

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

    // apply non-maximal supression to supress thick edges
    GrayImage* magSupressed = new GrayImage;
    edgenms(mag, orient, magSupressed);
    // writeGrayPPMImage(magSupressed, "magSupressed.ppm");

    // apply a threshold to get a binary image (255 is edge)
    GrayImage* magThreshold = new GrayImage;
    threshold(magSupressed, magThreshold, THRESHOLD);
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

    // apply non-maximal supression to supress thick edges 
    GrayImage* magSupressed = new GrayImage;
    edgenms(mag, orient, magSupressed);

    // apply a threshold to get a binary image (255 is edge)
    GrayImage* magThreshold = new GrayImage;
    threshold(magSupressed, magThreshold, THRESHOLD);
    // -------Reuse from processTemplate ends-------

    // initialize accumulator array (4D: Scale x Rotation x Width x Height)
    int width = magThreshold->width;
    int height = magThreshold->height;
    std::vector<std::vector<std::vector<std::vector<int>>>> accumulator(nScaleSlices, std::vector<std::vector<std::vector<int>>>(nRotationSlices, std::vector<std::vector<int>>(width, std::vector<int>(height, 0))));
    std::vector<std::vector<Point>> blockMaxima(width/blockSize+1, std::vector<Point>(height/blockSize+1, (struct Point){0, 0, 0, 0.f, 0.f}));

    // Each edge pixel vote
    printf("------Start calculating accumulator-------\n");
    int _max = 0;
    for (int j = 0 ; j < height; j++) {
        for (int i = 0 ; i < width; i++) {
            if (magThreshold->data[j * width + i] > 254) {
                // calculate edge gradient
                float phi = orient->data[j * width + i]; // gradient direction in [0,360)
                int iSlice = static_cast<int>(phi / deltaRotationAngle);
                std::vector<rEntry> entries = rTable[iSlice];
                for (auto entry: entries){
                    float r = sqrt(entry.x*entry.x + entry.y*entry.y);
                    float alpha = entry.alpha;
                    for (int is = 0; is < nScaleSlices; is++){
                        float s = is*deltaScaleRatio+MINSCALE;
                        for (int itheta = 0; itheta < nRotationSlices; itheta++){
                            float theta = itheta*deltaRotationAngle/180.f*PI;
                            int xc = i + round(r*s*cos(alpha+theta));
                            int yc = j + round(r*s*sin(alpha+theta));
                            if (xc>=0 && xc<width && yc>=0 && yc<height){
                                accumulator[is][itheta][xc][yc]++;
                                // find maximum for each block
                                if (accumulator[is][itheta][xc][yc] > blockMaxima[xc/blockSize][yc/blockSize].hits){
                                    blockMaxima[xc/blockSize][yc/blockSize].hits = accumulator[is][itheta][xc][yc];
                                    blockMaxima[xc/blockSize][yc/blockSize].x = xc;
                                    blockMaxima[xc/blockSize][yc/blockSize].y = yc;
                                    blockMaxima[xc/blockSize][yc/blockSize].scale = s;
                                    blockMaxima[xc/blockSize][yc/blockSize].rotation = theta;
                                    _max = (blockMaxima[xc/blockSize][yc/blockSize].hits > _max)? blockMaxima[xc/blockSize][yc/blockSize].hits: _max;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    printf("max value in accumulator: %d\n", _max);
    printf("------End calculating accumulator-------\n");

    // find local maxima
    int maximaThres = round(_max * thresRatio);
    for (int i = 0; i <= width/blockSize ; i++){
        for (int j = 0; j <= height/blockSize; j++){
            if (localMaxima(blockMaxima, i, j, maximaThres)){
                Point p = (struct Point){blockMaxima[i][j].x, blockMaxima[i][j].y, blockMaxima[i][j].hits, blockMaxima[i][j].scale, blockMaxima[i][j].rotation};
                hitPoints.push_back(p);
                printf("hit points: %d %d hits: %d scale: %f rotation: %f\n", p.x, p.y, p.hits, p.scale, p.rotation);
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

bool SeqGeneralHoughTransform::localMaxima(std::vector<std::vector<Point>> blockMaxima, int i, int j, int maximaThres){
    if (blockMaxima[i][j].hits < maximaThres) return false;
    int boundLeft = (i-1 > 0)? i-1 : 0;
    int boundRight = (i+1 < blockMaxima.size())? i+1 : blockMaxima.size()-1;
    int boundBottom = (j-1 > 0)? j-1 : 0;
    int boundTop = (j+1 < blockMaxima[0].size())? j+1 : blockMaxima[0].size()-1;
    for (int ii = boundLeft; ii <= boundRight ; ii++){
        for (int jj = boundBottom; jj <= boundTop; jj++){
            if (ii==i && jj==j) continue;
            if (blockMaxima[ii][jj].hits >= blockMaxima[i][j].hits){
                int xc0 = blockMaxima[i][j].x;
                int yc0 = blockMaxima[i][j].y;
                int xc1 = blockMaxima[ii][jj].x;
                int yc1 = blockMaxima[ii][jj].y;
                if ((abs(xc1 - xc0) < blockSize) || (abs(yc1 - yc0) < blockSize))
                    return false;
            } 
        }
    }
    return true;
}

void SeqGeneralHoughTransform::saveOutput() {
    const int size = 5;
    for (auto p : hitPoints) {
        int px = p.x;
        int py = p.y;
        for (int i = -size; i <= size; i++) {
            for (int j = -size; j <= size; j++) {
                int coloredPx = px + i;
                int coloredPy = py + j;
                if (coloredPx >= 0 && coloredPx < src->width && coloredPy >= 0 && coloredPy < src->height) {
                    int idx = coloredPy * src->width + coloredPx;
                    src->data[idx * 3] = 255;
                    src->data[idx * 3 + 1] = 0;
                    src->data[idx * 3 + 2] = 0;
                }
            }
        }
    }
    writePPMImage(src, "output.ppm");
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
            // do not consider image boundary
            if (j == 0 || j == source->height - 1 || i == 0 || i == source->width - 1) result->data[j * source->width + i] = 0;
            else result->data[j * source->width + i] = tmp;
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

bool keepPixel(const GrayImage* magnitude, int i, int j, int gradient) {
    int neighbourOnei = i;
    int neighbourOnej = j;
    int neighbourTwoi = i;
    int neighbourTwoj = j;
    
    switch (gradient) {
    case 0:
        neighbourOnei -= 1;
        neighbourTwoi += 1;
        break;
    case 45:
        neighbourOnej -= 1;
        neighbourOnei += 1;
        neighbourTwoj += 1;
        neighbourTwoi -= 1;
        break;
    case 90:
        neighbourOnej -= 1;
        neighbourTwoj += 1;
        break;
    default: // 135
        neighbourOnej -= 1;
        neighbourOnei -= 1;
        neighbourTwoj += 1;
        neighbourTwoi += 1;
    }
    
    float neighbourOne = magnitude->data[neighbourOnej * magnitude->width + neighbourOnei];
    float neighbourTwo = magnitude->data[neighbourTwoj * magnitude->width + neighbourTwoi];
    float cur = magnitude->data[j * magnitude->width + i];
    
    return (neighbourOne <= cur) && (neighbourTwo <= cur);
}

void SeqGeneralHoughTransform::edgenms(const GrayImage* magnitude, const GrayImage* orientation, GrayImage* result) {
    result->setGrayImage(orientation->width, orientation->height);
    for (int j = 0 ; j < orientation->height; j++) {
        for (int i = 0 ; i < orientation->width; i++) {
            int pixelGradient = static_cast<int>(orientation->data[j * orientation->width + i] / 45) * 45 % 180;
            if (keepPixel(magnitude, i, j, pixelGradient)) {
                result->data[j * orientation->width + i] = magnitude->data[j * orientation->width + i];
            } else {
                result->data[j * orientation->width + i] = 0;
            }
        }
    }
}

void SeqGeneralHoughTransform::threshold(const GrayImage* magnitude, GrayImage* result, int threshold) {
    result->setGrayImage(magnitude->width, magnitude->height);
    for (int i = 0; i < magnitude->width * magnitude->height; i++) {
        if (magnitude->data[i] > threshold) result->data[i] = 255;
        else result->data[i] = 0;
    }
}

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
