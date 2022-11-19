#include <string>
#include <iostream>
#include <cmath>

#include "cudaGeneralHoughTransform.h"
#include "utils.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

const int THRESHOLD = 200;
const float PI = 3.14159265;
const int nRotationSlices = 72;
const float deltaRotationAngle = 360 / nRotationSlices;
const float MAXSCALE = 1.4f;
const float MINSCALE = 0.6f;
const float deltaScaleRatio = 0.1f;
const int nScaleSlices = (MAXSCALE - MINSCALE) / deltaScaleRatio + 1;
const int blockSize = 10;
const float thresRatio = 0.9;

CudaGeneralHoughTransform::CudaGeneralHoughTransform() {
    tpl = new Image;
    src = new Image;
}

CudaGeneralHoughTransform::~CudaGeneralHoughTransform() {
    if (tpl) delete tpl;
    if (src) delete src;
}

void CudaGeneralHoughTransform::setup() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}

// 1. no need to allocate each image in cpu
// 2. convolve/magnitude/orientation/nms/threshold/createRTable: each thread -> pixel
// 3. parallel convolveX, convolveY
// 4. parallel magnitude & orientation
void CudaGeneralHoughTransform::processTemplate() {
    // convert template to gray image
    GrayImage* grayTpl = new GrayImage;
    convertToGray(tpl, grayTpl);

    // apply sobel_x -> gradient in x
    std::vector<std::vector<int>> sobelX = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    GrayImage* gradientX = new GrayImage;
    convolve(sobelX, grayTpl, gradientX);

    // apply soble_y -> gradient in y
    std::vector<std::vector<int>> sobelY = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
    GrayImage* gradientY = new GrayImage;
    convolve(sobelY, grayTpl, gradientY);

    // grascale magnitude = sqrt(square(x) + square(y))
    GrayImage* mag = new GrayImage;
    magnitude(gradientX, gradientY, mag);

    // grascale orientation = (np.degrees(np.arctan2(image_y,image_x))+360)%360
    // orientation in [0, 360)
    GrayImage* orient = new GrayImage;
    orientation(gradientX, gradientY, orient);

    // apply non-maximal supression to supress thick edges
    GrayImage* magSupressed = new GrayImage;
    edgenms(mag, orient, magSupressed);

    // apply a threshold to get a binary image (255 is edge)
    GrayImage* magThreshold = new GrayImage;
    threshold(magSupressed, magThreshold, THRESHOLD);

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

void CudaGeneralHoughTransform::accumulateSource() {
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
    std::vector<std::vector<std::vector<std::vector<int>>>> accumulator(nScaleSlices, std::vector<std::vector<std::vector<int>>>(nRotationSlices, std::vector<std::vector<int>>(width/blockSize+1, std::vector<int>(height/blockSize+1, 0))));
    std::vector<std::vector<Point>> blockMaxima(width/blockSize+1, std::vector<Point>(height/blockSize+1, (struct Point){0, 0, 0, 0.f, 0.f}));

    // Each edge pixel vote
    printf("------Start calculating accumulator-------\n");
    // int _max = 0;
    // for (int j = 0 ; j < height; j++) {
    //     for (int i = 0 ; i < width; i++) {
    //         if (magThreshold->data[j * width + i] > 254) {
    //             // calculate edge gradient
    //             float phi = orient->data[j * width + i]; // gradient direction in [0,360)
    //             for (int itheta = 0; itheta < nRotationSlices; itheta++){
    //                 float theta = itheta * deltaRotationAngle;
    //                 float theta_r = theta / 180.f * PI;

    //                 // minus mean rotate back by theta
    //                 int iSlice = static_cast<int>(fmod(phi - theta + 360, 360) / deltaRotationAngle);
    //                 std::vector<rEntry> entries = rTable[iSlice];
    //                 for (auto entry: entries){
    //                     float r = entry.r;
    //                     float alpha = entry.alpha;
    //                     for (int is = 0; is < nScaleSlices; is++){
    //                         float s = is * deltaScaleRatio + MINSCALE;
    //                         int xc = i + round(r * s * cos(alpha + theta_r));
    //                         int yc = j + round(r * s * sin(alpha + theta_r));
    //                         if (xc >= 0 && xc < width && yc >= 0 && yc < height){
    //                             // use block here? too fine grained for pixel level
    //                             accumulator[is][itheta][xc/blockSize][yc/blockSize]++;
    //                             // find maximum for each block
    //                             if (accumulator[is][itheta][xc/blockSize][yc/blockSize] > blockMaxima[xc/blockSize][yc/blockSize].hits){
    //                                 blockMaxima[xc/blockSize][yc/blockSize].hits = accumulator[is][itheta][xc/blockSize][yc/blockSize];
    //                                 blockMaxima[xc/blockSize][yc/blockSize].x = xc/blockSize * blockSize + blockSize / 2;
    //                                 blockMaxima[xc/blockSize][yc/blockSize].y = yc/blockSize * blockSize + blockSize / 2;
    //                                 blockMaxima[xc/blockSize][yc/blockSize].scale = s;
    //                                 blockMaxima[xc/blockSize][yc/blockSize].rotation = theta;
    //                                 _max = (blockMaxima[xc/blockSize][yc/blockSize].hits > _max)? blockMaxima[xc/blockSize][yc/blockSize].hits: _max;
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    printf("max value in accumulator: %d\n", _max);
    printf("------End calculating accumulator-------\n");

    // find local maxima
    // int maximaThres = round(_max * thresRatio);
    // for (int i = 0; i <= width/blockSize ; i++){
    //     for (int j = 0; j <= height/blockSize; j++){
    //         if (blockMaxima[i][j].hits > maximaThres) {
    //             Point p = (struct Point){blockMaxima[i][j].x, blockMaxima[i][j].y, blockMaxima[i][j].hits, blockMaxima[i][j].scale, blockMaxima[i][j].rotation};
    //             hitPoints.push_back(p);
    //             printf("hit points: %d %d hits: %d scale: %f rotation: %f\n", p.x, p.y, p.hits, p.scale, p.rotation);
    //         }
    //     }
    // }

    // find all edge points, put into a vector

    // (1) naive approach: 1D partition -> divergent control flow on entries

    // (2) better approach: 
    //     a. put edge points into buckets by phi
    //     b. points with the same phi go together in a kernel (1D) each block has the same phi value

    // (3) more parallelism: 
    //     a. put edge points into buckets by phi
    //     b. go by same phi, then same theta (2D) -> shared slice of 3D accumulator || then same scale (3D?) -> shared slice of 2D accumulator

    // Some high level notes: 
    // a. use atomic add when update accumulator
    // b. any idea to reduce global memory access?

    // memory deallocation
    delete graySrc;
    delete gradientX;
    delete gradientY;
    delete mag;
    delete orient;
    delete magThreshold;
}

void CudaGeneralHoughTransform::saveOutput() {
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
    writePPMImage(src, "output_cuda.ppm");
}

bool CudaGeneralHoughTransform::loadTemplate(std::string filename) {
    if (!readPPMImage(filename, tpl)) return false;
    return true;
}

bool CudaGeneralHoughTransform::loadSource(std::string filename) {
    if (!readPPMImage(filename, src)) return false;
    return true;
}

void CudaGeneralHoughTransform::convertToGray(const Image* image, GrayImage* result) {
    result->setGrayImage(image->width, image->height);
    for (int i = 0; i < image->width * image->height; i++) {
        result->data[i] = (image->data[3 * i] + image->data[3 * i + 1] + image->data[3 * i + 2]) / 3.f;
    }
}

void CudaGeneralHoughTransform::convolve(std::vector<std::vector<int>> filter, const GrayImage* source, GrayImage* result) {
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

void CudaGeneralHoughTransform::magnitude(const GrayImage* gradientX, const GrayImage* gradientY, GrayImage* result) {
    result->setGrayImage(gradientX->width, gradientX->height);
    for (int i = 0; i < gradientX->width * gradientX->height; i++) {
        result->data[i] = sqrt(gradientX->data[i] * gradientX->data[i] + gradientY->data[i] * gradientY->data[i]);
    }
}

void CudaGeneralHoughTransform::orientation(const GrayImage* gradientX, const GrayImage* gradientY, GrayImage* result) {
    result->setGrayImage(gradientX->width, gradientX->height);
    for (int i = 0; i < gradientX->width * gradientX->height; i++) {
        result->data[i] = fmod(atan2(gradientY->data[i], gradientX->data[i]) * 180 / PI + 360, 360);
    }
}

void CudaGeneralHoughTransform::edgenms(const GrayImage* magnitude, const GrayImage* orientation, GrayImage* result) {
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

void CudaGeneralHoughTransform::threshold(const GrayImage* magnitude, GrayImage* result, int threshold) {
    result->setGrayImage(magnitude->width, magnitude->height);
    for (int i = 0; i < magnitude->width * magnitude->height; i++) {
        if (magnitude->data[i] > threshold) result->data[i] = 255;
        else result->data[i] = 0;
    }
}

void CudaGeneralHoughTransform::createRTable(const GrayImage* orientation, const GrayImage* magThreshold) {
    rTable.clear();
    rTable.resize(nRotationSlices);

    centerX = orientation->width / 2;
    centerY = orientation->height / 2;

    for (int j = 0 ; j < orientation->height; j++) {
        for (int i = 0 ; i < orientation->width; i++) {
            if (magThreshold->data[j * orientation->width + i] > 254) {
                float phi = fmod(orientation->data[j * orientation->width + i], 360); // gradient direction in [0,360)
                int iSlice = static_cast<int>(phi / deltaRotationAngle);
                rEntry entry; 
                auto entryX = centerX - i;
                auto entryY = centerY - j;
                entry.r = sqrt(entryX * entryX + entryY * entryY);
                entry.alpha = static_cast<float>(atan2(entryY, entryX));
                rTable[iSlice].push_back(entry);
            }
        }
    }
}
