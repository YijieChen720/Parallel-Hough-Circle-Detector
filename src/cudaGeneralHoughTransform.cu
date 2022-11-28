#include <string>
#include <iostream>
#include <cmath>

#include "cudaGeneralHoughTransform.h"
#include "utils.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>

#define TPB_X 32
#define TPB_Y 32
#define TPB (TPB_X*TPB_Y)

struct GlobalConstants {
    int THRESHOLD;
    float PI;
    int nRotationSlices;
    float deltaRotationAngle;
    float MAXSCALE;
    float MINSCALE;
    float deltaScaleRatio;
    int nScaleSlices;
    int blockSize;
    float thresRatio;
};

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

__constant__ GlobalConstants cuConstParams;
__device__ float* grayData;
__device__ float* orient;
__device__ rEntry* entries;
__device__ int* startPos;

struct compare_entry_by_rotation
{
    __host__ __device__
    bool operator()(const rEntry &r1, const rEntry &r2)
    {
        return r1.iSlice < r2.iSlice;
    }
};

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

    GlobalConstants params;
    params.THRESHOLD = THRESHOLD;
    params.PI = PI;
    params.nRotationSlices = nRotationSlices;
    params.deltaRotationAngle = deltaRotationAngle;
    params.MAXSCALE = MAXSCALE;
    params.MINSCALE = MINSCALE;
    params.deltaScaleRatio = deltaScaleRatio;
    params.nScaleSlices = nScaleSlices;
    params.blockSize = blockSize;
    params.thresRatio = thresRatio;

    cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));

}

__device__ __inline__
void convolve(int filter[3][3], float* source, float* result, int width, int height, int i, int j, int localIdx) {
    // if (i == 0 && j == 0 && (filter.size() != 3 || filter[0].size() != 3)) {
    //     std::cerr << "ERROR: convolve() only supports 3x3 filter.\n";
    //     return;
    // }
    float tmp = 0.f;
    for (int jj = -1; jj <= 1; jj++) {
        for (int ii = -1; ii <= 1; ii++) {
            int row = j + jj;
            int col = i + ii;
            if (row < 0 || row >= height || col < 0 || col >= width) {
                // out of image bound, do nothing
            } else {
                int idx = row * width + col;
                tmp += source[idx] * filter[jj + 1][ii + 1];
            }
        }
    }
    // do not consider image boundary
    if (j == 0 || j == height - 1 || i == 0 || i == width - 1) result[localIdx] = 0;
    else result[localIdx] = tmp;
}

__device__ __inline__
void magnitude(const float* gradientX, const float* gradientY, float* result, int i, int localIdx) {
    grayData[i] = sqrtf(gradientX[localIdx] * gradientX[localIdx] + gradientY[localIdx] * gradientY[localIdx]);
}

__device__ __inline__
void orientation(const float* gradientX, const float* gradientY, float* result, int i, int localIdx) {
    result[i] = fmodf(atan2f(gradientY[localIdx], gradientX[localIdx]) * 180 / cuConstParams.PI + 360, 360);
}

__device__ __inline__
bool keepPixel(const float* magnitude, int indexX, int indexY, int width, int height, int gradient) {
    int neighbourOnei = threadIdx.x;
    int neighbourOnej = threadIdx.y;
    int neighbourTwoi = threadIdx.x;
    int neighbourTwoj = threadIdx.y;
    
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
    
    float neighbourOne, neighbourTwo;
    // out of the bound of this block => neighbour's pixel => access global memory
    if (neighbourOnei < 0 || neighbourOnei >= TPB_X || neighbourOnej < 0 || neighbourOnej >= TPB_Y){
        neighbourOne = grayData[(indexY + neighbourOnej) * width + (indexX + neighbourOnei)];
    }
    // in the bound of this block => access shared memory
    else{
        neighbourOne = magnitude[neighbourOnej * TPB_X + neighbourOnei];
    }
    // out of the bound of this block => neighbour's pixel => access global memory
    if (neighbourTwoi < 0 || neighbourTwoi >= TPB_X || neighbourTwoj < 0 || neighbourTwoj >= TPB_Y){
        neighbourTwo = grayData[(indexY + neighbourOnej) * width + (indexX + neighbourOnei)];
    }
    // in the bound of this block => access shared memory
    else{
        neighbourTwo = magnitude[neighbourTwoj * TPB_X + neighbourTwoi];
    }
    float cur = magnitude[threadIdx.y * TPB_X + threadIdx.x];
    
    return (neighbourOne <= cur) && (neighbourTwo <= cur);
}

__device__ __inline__
void edgenms(float* magnitude, float* orientation, float* result, int width, int height, int i, int j, int localIdx) {
    int pixelGradient = static_cast<int>(orientation[localIdx] / 45) * 45 % 180;
    if (keepPixel(magnitude, i, j, width, height, pixelGradient)) {
        result[localIdx] = magnitude[localIdx];
    } else {
        result[localIdx] = 0;
    }
}

__device__ __inline__
void threshold(float* magnitude, float* result, int threshold, int index, int localIdx) {
    if (magnitude[localIdx] > threshold) result[localIdx] = index;
    else result[localIdx] = -1;
}

__device__ __inline__
void createRTable(const float* orient, const float* magThreshold, int width, int height, int i, int j, int index, int localIdx) {
    int centerX = width / 2;
    int centerY = height / 2;

    if (magThreshold[localIdx] >= 0) {
        float phi = fmodf(orient[localIdx], 360); // gradient direction in [0,360)
        entries[index].iSlice = static_cast<int>(phi / cuConstParams.deltaRotationAngle);
        int entryX = centerX - i;
        int entryY = centerY - j;
        entries[index].r = sqrtf(entryX * entryX + entryY * entryY);
        entries[index].alpha = static_cast<float>(atan2f(entryY, entryX));
    }
    else entries[index].iSlice = -1;

}

__global__ void kernelConvertToGray(float* data, int width, int height){
    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;
    int index = indexY * width + indexX;

    if (indexX >= width || indexY >= height) return;

    data[index] = (grayData[3 * index] + grayData[3 * index + 1] +grayData[3 * index + 2]) / 3.f;
}

// parallel convolveX, convolveY
// parallel magnitude, orientaion
// __global__ void kernelProcessStep1Updated(int width, int height){
//     int indexX = blockIdx.x * blockDim.x + threadIdx.x;
//     int indexY = blockIdx.y * blockDim.y + threadIdx.y;
//     int index = (indexY * blockDim.x * TPB_X + indexX)/2;
//     indexX = index % width;
//     indexY = index / width;
//     int localIdx = threadIdx.y * TPB_X + threadIdx.x;

//     if (indexX >= width || indexY >= height) return;

//     __shared__ float gradientX[TPB/2];
//     __shared__ float gradientY[TPB/2];
//     int sobel[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
//     if (localIdx%2==0){
//         convolve(sobel, grayData, gradientX, width, height, indexX, indexY);
//     }
//     else{
//         convolve(sobel, grayData, gradientY, width, height, indexX, indexY);
//     }
//     __syncthreads();

//     if (localIdx%2==0){
//         magnitude(gradientX, gradientY, grayData, index, localIdx/2);
//     }
//     else{
//         orientation(gradientX, gradientY, orient, index, localIdx/2);
//     }
// }


__global__ void kernelProcessStep1(int width, int height){
    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;
    int index = indexY * width + indexX;
    int localIdx = threadIdx.y*TPB_X+threadIdx.x;

    if (indexX >= width || indexY >= height) return;

    __shared__ float gradientX[TPB];
    __shared__ float gradientY[TPB];
    int sobel[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    convolve(sobel, grayData, gradientX, width, height, indexX, indexY, localIdx);
    convolve(sobel, grayData, gradientY, width, height, indexX, indexY, localIdx);

    magnitude(gradientX, gradientY, grayData, index, localIdx);
    orientation(gradientX, gradientY, orient, index, localIdx);
}

__global__ void kernelProcessStep2(int width, int height, bool tpl){
    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;
    int index = indexY * width + indexX;
    int localIdx = threadIdx.y * TPB_X + threadIdx.x;

    if (indexX >= width || indexY >= height) return;

    __shared__ float magnitude[TPB]; 
    __shared__ float orientation[TPB]; 
    __shared__ float magThreshold[TPB];
    // load this block's magnitude and orient value in shared memory, 
    // decreasing the global memory access
    magnitude[localIdx] = grayData[index]; 
    orientation[localIdx] = orient[index]; 
    __syncthreads();
    
    edgenms(magnitude, orientation, magThreshold, width, height, indexX, indexY, localIdx); 
    threshold(magThreshold, magThreshold, cuConstParams.THRESHOLD, index, localIdx);

    // if processing template, only need to create R table
    if (tpl){
        createRTable(orientation, magThreshold, width, height, indexX, indexY, index, localIdx);
    }
    // if processing source, write the result data back to global memory
    else{
        grayData[index] = magThreshold[localIdx];
    }
}

__global__ void processRTable(int width, int height){
    int i = 0;
    int angle = cuConstParams.deltaRotationAngle;
    while(i < width * height){
        if (entries[i].iSlice < 0) continue;
        while (i!=0 || entries[i].iSlice == entries[i-1].iSlice) {i++;}
        for (int j = entries[i-1].iSlice/angle+1; j < entries[i].iSlice/angle+1; j++){
            startPos[j] = i;
        }
    }    
}

// 1. no need to allocate each image in cpu
// 2. convolve/magnitude/orientation/nms/threshold/createRTable: each thread -> pixel
// 3. parallel convolveX, convolveY
// 4. parallel magnitude & orientation
void CudaGeneralHoughTransform::processTemplate() {
    float* deviceTplData;
    float* tplGrayData;
    float* tmpOrientation;
    rEntry* tmpEntries;

    cudaMalloc(&deviceTplData, tpl->width * tpl->height * sizeof(float));
    cudaMalloc(&tplGrayData, 3 * tpl->width * tpl->height * sizeof(float));
    cudaMemcpy(deviceTplData, tpl->data, tpl->width * tpl->height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(grayData, &tplGrayData, sizeof(float*));

    cudaMalloc(&tmpEntries, tpl->width * tpl->height * sizeof(rEntry));
    cudaMemcpyToSymbol(entries, &tmpEntries, sizeof(rEntry*));
    
    cudaMalloc(&tmpOrientation, tpl->width * tpl->height * sizeof(float));
    cudaMemcpyToSymbol(orient, &tmpOrientation, sizeof(float*));

    dim3 blockDim(TPB_X, TPB_Y, 1);
    dim3 gridDim((tpl->width + TPB_X - 1) / TPB_X, 
                 (tpl->height + TPB_Y - 1) / TPB_Y, 1);
    
    kernelConvertToGray<<<gridDim, blockDim>>>(deviceTplData, tpl->width, tpl->height);
    cudaDeviceSynchronize();

    kernelProcessStep1<<<gridDim, blockDim>>>(tpl->width, tpl->height);
    cudaDeviceSynchronize();

    kernelProcessStep2<<<gridDim, blockDim>>>(tpl->width, tpl->height, true);
    cudaDeviceSynchronize();  

    thrust::device_ptr<rEntry> entriesThrust = thrust::device_pointer_cast(tmpEntries); 

    thrust::sort(entriesThrust, entriesThrust + tpl->width * tpl->height, compare_entry_by_rotation());
    // cudaMemcpyToSymbol(entries, &tmpEntries, sizeof(rEntry*));

    int* tmpStartPos;
    cudaMalloc(&tmpStartPos, nRotationSlices * sizeof(int));
    cudaMemcpyToSymbol(startPos, &tmpStartPos, sizeof(int*));
    processRTable<<<1, 1>>>(tpl->width, tpl->height);
    
    // memory deallocation
    cudaFree(tmpEntries);
    cudaFree(deviceTplData);
    cudaFree(tmpOrientation);
    cudaFree(tplGrayData);
}

void CudaGeneralHoughTransform::accumulateSource() {
    float* deviceSrcData;
    float* srcGrayData;
    float* tmpOrientation;

    cudaMalloc(&deviceSrcData, src->width * src->height * sizeof(float));
    cudaMalloc(&srcGrayData, 3 * src->width * src->height * sizeof(float));
    cudaMemcpy(deviceSrcData, src->data, src->width * src->height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(grayData, &srcGrayData, sizeof(float*));

    cudaMalloc(&tmpOrientation, src->width * src->height * sizeof(float));
    cudaMemcpyToSymbol(orient, &tmpOrientation, sizeof(float*));

    dim3 blockDim(TPB_X, TPB_Y, 1);
    dim3 gridDim((src->width + TPB_X - 1) / TPB_X, 
                 (src->height + TPB_Y - 1) / TPB_Y, 1);
    
    kernelConvertToGray<<<gridDim, blockDim>>>(deviceSrcData, src->width, src->height);
    cudaDeviceSynchronize();

    kernelProcessStep1<<<gridDim, blockDim>>>(src->width, src->height);
    cudaDeviceSynchronize();

    kernelProcessStep2<<<gridDim, blockDim>>>(src->width, src->height, false);
    cudaDeviceSynchronize();  

    // GrayImage* magThreshold = new GrayImage;
    // cudaMemcpyFromSymbol(magThreshold->data, grayData, src->width * src->height * sizeof(float), cudaMemcpyDeviceToHost);

    // memory deallocation
    cudaFree(srcGrayData);
    cudaFree(tmpOrientation);
    cudaFree(deviceSrcData);
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

