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

__constant__ GlobalConstants cuConstParams;

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

    // cudaMalloc(&cudaDeviceTpl, sizeof(Image));
    // cudaMalloc(&cudaDeviceSrc, sizeof(Image));

    // cudaMemcpy(cudaDeviceTpl, tpl, sizeof(Image), cudaMemcpyHostToDevice);
    // cudaMemcpy(cudaDeviceSrc, src, sizeof(Image), cudaMemcpyHostToDevice);

    GlobalConstants params;
    params.THRESHOLD = 200;
    params.PI = 3.14159265;
    params.nRotationSlices = 72;
    params.deltaRotationAngle = 360 / nRotationSlices;
    params.MAXSCALE = 1.4f;
    params.MINSCALE = 0.6f;
    params.deltaScaleRatio = 0.1f;
    params.nScaleSlices = (MAXSCALE - MINSCALE) / deltaScaleRatio + 1;
    params.blockSize = 10;
    params.thresRatio = 0.9;

    cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));

}

__global__ void kernelConvertToGray(int width, int height){
    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;
    int index = indexY * blockDim.x * TPB_X + indexX;

    if (indexX >= width || indexY >= height) return;
    deviceData[index] = (deviceData[3 * index] + deviceData[3 * index + 1] +deviceData[3 * index + 2]) / 3.f;
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
//         convolve(sobel, deviceData, gradientX, width, height, indexX, indexY);
//     }
//     else{
//         convolve(sobel, deviceData, gradientY, width, height, indexX, indexY);
//     }
//     __syncthreads();

//     if (localIdx%2==0){
//         magnitude(gradientX, gradientY, deviceData, index, localIdx/2);
//     }
//     else{
//         orientation(gradientX, gradientY, orient, index, localIdx/2);
//     }
// }


__global__ void kernelProcessStep1(int width, int height){
    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;
    int index = indexY * blockDim.x * TPB_X + indexX;
    int localIdx = threadIdx.y*TPB_X+threadIdx.x;

    if (indexX >= width || indexY >= height) return;

    __shared__ float gradientX[TPB];
    __shared__ float gradientY[TPB];
    int sobel[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    convolve(sobel, deviceData, gradientX, width, height, indexX, indexY);
    convolve(sobel, deviceData, gradientY, width, height, indexX, indexY);

    magnitude(gradientX, gradientY, deviceData, index, localIdx);
    orientation(gradientX, gradientY, orient, index, localIdx);
}

__global__ void kernelProcessStep2(int width, int height, bool tpl){
    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;
    int index = indexY * blockDim.x * TPB_X + indexX;
    int localIdx = threadIdx.y * TPB_X + threadIdx.x;

    if (indexX >= width || indexY >= height) return;

    // __shared__ float result[TPB];
    __shared__ float magnitude[TPB]; 
    __shared__ float orient[TPB]; 
    // load this block's magnitude and orient value in shared memory, 
    // decreasing the global memory access
    magnitude[localIdx] = deviceData[index]; 
    orient[localIdx] = orient[index]; 
    
    edgenms(magnitude, orient, magnitude, width, height, indexX, indexY, localIdx); 
    threshold(magnitude, magnitude, cuConstParam.THRESHOLD, index, localIdx);

    // if processing template, only need to create R table
    if (tpl){
        createRTable(orient, magnitude, width, height, index);
    }
    // if processing source, write the result data back to global memory
    else{
        deviceData[index] = magnitude[localIdx];
    }
}

__global__ void processRTable(int width, int height){
    int i = 0;
    int angle = cuConstParam.deltaRotationAngle;
    while(i < width * height){
        if (entries[i] < 0) continue;
        while (i!=0 || entries[i] == entries[i-1]) {i++};
        for (j = entries[i-1]/angle+1; j < entries[i]/angle+1; j+=angle){
            startPos[j] = i;
        }
    }    
}

// 1. no need to allocate each image in cpu
// 2. convolve/magnitude/orientation/nms/threshold/createRTable: each thread -> pixel
// 3. parallel convolveX, convolveY
// 4. parallel magnitude & orientation
void CudaGeneralHoughTransform::processTemplate() {
    float* deviceData;
    float* orient;
    rEntry* entries;

    cudaMalloc(&deviceData, tpl->width * tpl->height * sizeof(float));
    cudaMemcpy(deviceData, tpl->data, tpl->width * tpl->height * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&entries, tpl->width * tpl->height * sizeof(rEntry));
    
    cudaMalloc(&orient, tpl->width * tpl->height * sizeof(float));

    dim3 blockDim(TPB_X, TPB_Y, 1);
    dim3 gridDim((tpl->width + TPB_X - 1) / TPB_X, 
                 (tpl->height + TPB_Y - 1) / TPB_Y, 1);
    
    kernelConvertToGray<<<gridDim, blockDim>>>(tpl->width, tpl->height);
    cudaDeviceSynchronize();

    kernelProcessStep1<<<gridDim, blockDim>>>(tpl->width, tpl->height);
    cudaDeviceSynchronize();

    kernelProcessStep2<<<gridDim, blockDim>>>(tpl->width, tpl->height, true);
    cudaDeviceSynchronize();  

    thrust::device_ptr<rEntry> entriesThrust = thrust::device_pointer_cast(entriesThrust); 
    
    thrust::sort(entriesThrust, entriesThrust + tpl->width * tpl->height, compare_entry_by_rotation(entries));

    int* startPos;
    cudaMalloc(&startPos, params.nRotationSlices * sizeof(int));
    

    // memory deallocation
    cudaFree(entries);
    cudaFree(deviceData);
    cudaFree(orient);
}

void CudaGeneralHoughTransform::accumulateSource() {
    float* deviceData;
    float* orient;

    cudaMalloc(&deviceData, src->width * src->height * sizeof(float));
    cudaMemcpy(deviceData, src->data, src->width * src->height * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&orient, src->width * src->height * sizeof(float));

    dim3 blockDim(TPB_X, TPB_Y, 1);
    dim3 gridDim((src->width + TPB_X - 1) / TPB_X, 
                 (src->height + TPB_Y - 1) / TPB_Y, 1);
    
    kernelConvertToGray<<<gridDim, blockDim>>>(src->width, src->height);
    cudaDeviceSynchronize();

    kernelProcessStep1<<<gridDim, blockDim>>>(src->width, src->height);
    cudaDeviceSynchronize();

    kernelProcessStep2<<<gridDim, blockDim>>>(src->width, src->height, false);
    cudaDeviceSynchronize();  

    GrayImage* magThreshold = new GrayImage;
    cudaMemcpy(magThreshold->data, deviceData, src->width * src->height * sizeof(float), cudaMemcpyDeviceToHost);

    // memory deallocation
    cudaFree(deviceData);
    cudaFree(orient);
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

__global__ void convolve(int** filter, float* source, float* result, int width, int height, int i, int j) {
    if (index == 0 && (filter.size() != 3 || filter[0].size() != 3)) {
        std::cerr << "ERROR: convolve() only supports 3x3 filter.\n";
        return;
    }
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
    if (j == 0 || j == height - 1 || i == 0 || i == width - 1) result[threadIdx.y * TPB_X + threadIdx.x] = 0;
    else result[threadIdx.y * TPB_X + threadIdx.x] = tmp;
}

__global__ void magnitude(const float* gradientX, const float* gradientY, float* result, int i, int localIdx) {
    result[i] = sqrt(gradientX[localIdx] * gradientX[localIdx] + gradientY[localIdx] * gradientY[localIdx]);
}

__global__ void orientation(const float* gradientX, const float* gradientY, float* result, int i, int localIdx) {
    result[i] = fmod(atan2(gradientY[localIdx], gradientX[localIdx]) * 180 / cuConstParam.PI + 360, 360);
}

__global__ void edgenms(float* magnitude, float* orientation, float* result, int width, int height, int i, int j, int localIdx) {
    int pixelGradient = static_cast<int>(orientation[localIdx] / 45) * 45 % 180;
    if (keepPixel(magnitude, i, j, width, height, pixelGradient)) {
        result[localIdx] = magnitude[localIdx];
    } else {
        result[localIdx] = 0;
    }
}

__global__ void threshold(float* magnitude, float* result, int threshold, int index, int localIdx) {
    if (magnitude[localIdx] > threshold) result[localIdx] = index;
    else result[localIdx] = -1;
}

__global__ void createRTable(const float* orient, const float* magThreshold, int width, int height, int index) {
    centerX = width / 2;
    centerY = height / 2;

    if (magThreshold[localIdx] >= 0) {
        float phi = fmod(orient[localIdx], 360); // gradient direction in [0,360)
        int iSlice[index] = static_cast<int>(phi / cuConstParam.deltaRotationAngle);
        int entryX = centerX - i;
        int entryY = centerY - j;
        entryR[index] = sqrt(entryX * entryX + entryY * entryY);
        entryAlpha[index] = static_cast<float>(atan2(entryY, entryX));
    }
    else iSlice[index] = -1;

}

__global__ bool keepPixel(const float* magnitude, int indexX, int indexY, int width, int height, int gradient) {
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
    
    // out of the bound of this block => neighbour's pixel => access global memory
    if (neighbourOnei < 0 || neighbourOnei >= TPB_X || neighbourOnej < 0 || neighbourOnej >= TPB_Y){
        float neighbourOne = deviceData[(indexY + neighbourOnej) * width + (indexX + neighbourOnei)];
    }
    // in the bound of this block => access shared memory
    else{
        float neighbourOne = magnitude[neighbourOnej * TPB_X + neighbourOnei];
    }
    // out of the bound of this block => neighbour's pixel => access global memory
    if (neighbourTwoi < 0 || neighbourTwoi >= TPB_X || neighbourTwoj < 0 || neighbourTwoj >= TPB_Y){
        float neighbourTwo = deviceData[(indexY + neighbourOnej) * width + (indexX + neighbourOnei)];
    }
    // in the bound of this block => access shared memory
    else{
        float neighbourTwo = magnitude[neighbourTwoj * TPB_X + neighbourTwoi];
    }
    float cur = magnitude[threadIdx.y * TPB_X + threadIdx.x];
    
    return (neighbourOne <= cur) && (neighbourTwo <= cur);
}

struct compare_entry_by_rotation
{
    rEntry* entries;
    compare_entry_by_rotation(rEntry* entries) : entries(entry) {};
    __host__ __device__
    bool operator()(rEntry r1, rEntry r2)
    {
        int iSlice1 = r1.iSlice;
        int iSlice2 = r2.iSlice;
        return iSlice1 < iSlice2;
    }
};