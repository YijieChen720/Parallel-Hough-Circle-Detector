#include <string>
#include <iostream>
#include <cmath>

#include "cudaGeneralHoughTransform.h"
#include "utils.h"
#include "cycleTimer.h"

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

GlobalConstants params;
__constant__ GlobalConstants cuConstParams;

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans)  cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{   
    if (code != cudaSuccess)
    {
       fprintf(stderr, "CUDA Error: %s at %s:%d\n",
         cudaGetErrorString(code), file, line);
       if (abort) exit(code);
    }
}
#else
#define cudaCheckError(ans) ans
#endif

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

    params.THRESHOLD = 200;
    params.PI = 3.14159265;
    params.nRotationSlices = 72;
    params.deltaRotationAngle = 360 / params.nRotationSlices;
    params.MAXSCALE = 1.4f;
    params.MINSCALE = 0.6f;
    params.deltaScaleRatio = 0.1f;
    params.nScaleSlices = (params.MAXSCALE - params.MINSCALE) / params.deltaScaleRatio + 1;
    params.blockSize = 10;
    params.thresRatio = 0.3;

    cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));

}

__device__ __inline__
void convolve(int filter[3][3], const float* source, float* result, int width, int height, int i, int j, int localIdx) {
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
    result[i] = sqrtf(gradientX[localIdx] * gradientX[localIdx] + gradientY[localIdx] * gradientY[localIdx]);
}

__device__ __inline__
void orientation(const float* gradientX, const float* gradientY, float* result, int i, int localIdx) {
    result[i] = fmodf(atan2f(gradientY[localIdx], gradientX[localIdx]) * 180 / cuConstParams.PI + 360, 360);
}

__device__ __inline__
bool keepPixel(const float* mag, const float* magnitude, int indexX, int indexY, int width, int height, int gradient) {
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
        neighbourOne = mag[(indexY + neighbourOnej) * width + (indexX + neighbourOnei)];
    }
    // in the bound of this block => access shared memory
    else{
        neighbourOne = magnitude[neighbourOnej * TPB_X + neighbourOnei];
    }
    // out of the bound of this block => neighbour's pixel => access global memory
    if (neighbourTwoi < 0 || neighbourTwoi >= TPB_X || neighbourTwoj < 0 || neighbourTwoj >= TPB_Y){
        neighbourTwo = mag[(indexY + neighbourOnej) * width + (indexX + neighbourOnei)];
    }
    // in the bound of this block => access shared memory
    else{
        neighbourTwo = magnitude[neighbourTwoj * TPB_X + neighbourTwoi];
    }
    float cur = magnitude[threadIdx.y * TPB_X + threadIdx.x];
    
    return (neighbourOne <= cur) && (neighbourTwo <= cur);
}

__device__ __inline__
void edgenms(float* mag, float* magnitude, float* orientation, float* result, int width, int height, int i, int j, int localIdx) {
    int pixelGradient = static_cast<int>(orientation[localIdx] / 45) * 45 % 180;
    if (keepPixel(mag, magnitude, i, j, width, height, pixelGradient)) {
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
void createRTable(rEntry* entries, const float* orient, const float* magThreshold, int width, int height, int i, int j, int index, int localIdx) {
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

__global__ void kernelConvertToGray(const unsigned char* data, float* grayData, int width, int height){
    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;
    int index = indexY * width + indexX;

    if (indexX >= width || indexY >= height) return;

    grayData[index] = (data[3 * index] + data[3 * index + 1] + data[3 * index + 2]) / 3.f;
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

// __global__ void kernelConvolve(float* grayData, float* gradientX, float* gradientY, int width, int height){
//     int indexX = blockIdx.x * blockDim.x + threadIdx.x;
//     int indexY = blockIdx.y * blockDim.y + threadIdx.y;
//     int index = indexY * width + indexX;
//     int localIdx = threadIdx.y*TPB_X+threadIdx.x;

//     if (indexX >= width || indexY >= height) return;

//     // __shared__ float gradientX[TPB];
//     // __shared__ float gradientY[TPB];
//     int sobelX[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
//     int sobelY[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
//     convolve(sobelX, grayData, gradientX, width, height, indexX, indexY, index);
//     convolve(sobelY, grayData, gradientY, width, height, indexX, indexY, index); 
// }

// __global__ void kernelProcessStep1(float* gradientX, float* gradientY, float* mag, float* orient, int width, int height){
//     int indexX = blockIdx.x * blockDim.x + threadIdx.x;
//     int indexY = blockIdx.y * blockDim.y + threadIdx.y;
//     int index = indexY * width + indexX;
//     int localIdx = threadIdx.y*TPB_X+threadIdx.x;

//     if (indexX >= width || indexY >= height) return;
//     magnitude(gradientX, gradientY, mag, index, index);
//     orientation(gradientX, gradientY, orient, index, index);
// }

__global__ void kernelProcessStep1(float* grayData, float* mag, float* orient, int width, int height){
    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;
    int index = indexY * width + indexX;
    int localIdx = threadIdx.y*TPB_X+threadIdx.x;

    if (indexX >= width || indexY >= height) return;

    __shared__ float gradientX[TPB];
    __shared__ float gradientY[TPB];
    int sobelX[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    int sobelY[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
    convolve(sobelX, grayData, gradientX, width, height, indexX, indexY, localIdx);
    convolve(sobelY, grayData, gradientY, width, height, indexX, indexY, localIdx); 

    magnitude(gradientX, gradientY, mag, index, localIdx);
    orientation(gradientX, gradientY, orient, index, localIdx);
}

__global__ void kernelProcessStep2(float* mag, float* orient, rEntry* entries, int width, int height, bool tpl){
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
    magnitude[localIdx] = mag[index]; 
    orientation[localIdx] = orient[index]; 
    __syncthreads();
    
    edgenms(mag, magnitude, orientation, magThreshold, width, height, indexX, indexY, localIdx); 
    threshold(magThreshold, magThreshold, cuConstParams.THRESHOLD, index, localIdx);

    // if processing template, only need to create R table
    if (tpl){
        createRTable(entries, orientation, magThreshold, width, height, indexX, indexY, index, localIdx);
    }
    // if processing source, write the result data back to global memory
    else{
        mag[index] = magThreshold[localIdx];
    }
}

__global__ void processRTable(rEntry* entries, int* startPos, int width, int height){
    int i = 0;
    for (i = 0; i < width * height; i++){
        if (entries[i].iSlice >= 0) break;
    }
    while(i < width * height){
        if (i==0) {
            for (int j = 0; j < entries[0].iSlice+1; j++){
                startPos[j] = i;
            }
            i++;
            continue;
        }
        while (i < width * height && entries[i].iSlice == entries[i-1].iSlice) {i++;}
        for (int j = entries[i-1].iSlice+1; j < entries[i].iSlice+1; j++){
            startPos[j] = i;
        }
        i++;
    }
    for (int j = entries[i-2].iSlice+1; j < cuConstParams.nRotationSlices; j++){
        startPos[j] = width * height;
    }
    printf("startPos: %d %d %d %d %d\n", startPos[0], startPos[1], startPos[2], startPos[3], startPos[4]);
    printf("startPos: %d %d\n", startPos[70], startPos[71]);
}

struct is_edge
{
    __host__ __device__
    bool operator() (float x)
    {
        return (x >= 0);
    }
};

struct compare_point
{
    __host__ __device__
    bool operator()(const Point &p1, const Point &p2)
    {
        return p1.hits < p2.hits;
    }
};

struct filter_point 
{
    int threshold;
    filter_point(int thr) : threshold(thr) {};
    __host__ __device__
    bool operator()(const Point &p){
        return (p.hits > threshold);
    }
};

struct compare_pixel_by_orient
{
    float* srcOrient;
    compare_pixel_by_orient(float* orient) : srcOrient(orient) {};
    __host__ __device__
    bool operator()(float p1, float p2)
    {
        int idx1 = static_cast<int>(p1);
        int idx2 = static_cast<int>(p2);
        return srcOrient[idx1] < srcOrient[idx2];
    }
};

__global__ void accumulate_kernel_naive(int* accumulator, float* edgePixels, int numEdgePixels, float* srcOrient, int width, int height, int wblock, int hblock, rEntry* entries, int* startPos) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numEdgePixels) return;

    int pixelIndex = static_cast<int>(edgePixels[index]);
    int row = pixelIndex / width;
    int col = pixelIndex % width;
    
    float phi = srcOrient[pixelIndex]; // gradient direction in [0,360)
    for (int itheta = 0; itheta < cuConstParams.nRotationSlices; itheta++){
        float theta = itheta * cuConstParams.deltaRotationAngle;
        float theta_r = theta / 180.f * cuConstParams.PI;

        // minus mean rotate back by theta
        int iSlice = static_cast<int>(fmodf(phi - theta + 360, 360) / cuConstParams.deltaRotationAngle);
        
        // access RTable and traverse all entries
        int startIdx = startPos[iSlice];
        int endIdx;
        if (iSlice + 1 == cuConstParams.nRotationSlices) endIdx = width * height;
        else endIdx = startPos[iSlice + 1];
        for (int idx = startIdx; idx < endIdx; idx++) {
            rEntry entry = entries[idx];
            float r = entry.r;
            float alpha = entry.alpha;
            for (int is = 0; is < cuConstParams.nScaleSlices; is++){
                float s = is * cuConstParams.deltaScaleRatio + cuConstParams.MINSCALE;
                int xc = col + round(r * s * cos(alpha + theta_r));
                int yc = row + round(r * s * sin(alpha + theta_r));
                if (xc >= 0 && xc < width && yc >= 0 && yc < height) {
                    int accumulatorIndex = is * cuConstParams.nRotationSlices * hblock * wblock
                                           + itheta * hblock * wblock
                                           + yc / cuConstParams.blockSize * wblock
                                           + xc / cuConstParams.blockSize;
                    atomicAdd(accumulator + accumulatorIndex, 1);
                }
            }
        }
    }
}

__global__ void findCutoffs_kernel(float* edgePixels, int numEdgePixels, float* srcOrient, int threadsPerBlock, int* startIndex, int* numBlocksDevice) {
    int cur = 0;
    startIndex[0] = 0;
    *numBlocksDevice = 0;
    for (int i = 0; i < numEdgePixels; i++) {
        int pixelIndex = static_cast<int>(edgePixels[i]);
        float phi = srcOrient[pixelIndex]; // gradient direction in [0,360)
        int iSlice = static_cast<int>(phi / cuConstParams.deltaRotationAngle);
        if (iSlice > cur) {
            for (int j = cur + 1; j <= iSlice; j++) {
                startIndex[j] = i;
            }
            *numBlocksDevice += (startIndex[cur + 1] - startIndex[cur] + threadsPerBlock - 1) / threadsPerBlock;
            cur = iSlice;
        }
    }
    *numBlocksDevice += (numEdgePixels - startIndex[cur] + threadsPerBlock - 1) / threadsPerBlock;
    if (cur < cuConstParams.nRotationSlices - 1) {
        for (int i = cur + 1; i < cuConstParams.nRotationSlices; i++) {
            startIndex[i] = numEdgePixels;
        }
    }
}

__global__ void fillIntervals_kernel(int* startIndex, int numEdgePixels, int threadsPerBlock, int* blockStarts, int* blockEnds) {
    int cur = 0;
    int idx = 0;
    int curIdx = startIndex[cur];
    while (cur < cuConstParams.nRotationSlices - 1) {
        if (curIdx + threadsPerBlock < startIndex[cur + 1]) {
            blockStarts[idx] = curIdx;
            blockEnds[idx] = curIdx + threadsPerBlock - 1;
            curIdx += threadsPerBlock;
            idx++;
        } else if (startIndex[cur] != startIndex[cur + 1]) {
            blockStarts[idx] = curIdx;
            blockEnds[idx] = startIndex[cur + 1] - 1;
            idx++;
            cur++;
            curIdx = startIndex[cur];
        } else {
            cur++;
        }
    }
    while (curIdx < numEdgePixels) {
        blockStarts[idx] = curIdx;
        blockEnds[idx] = min(curIdx + threadsPerBlock - 1, numEdgePixels - 1);
        idx++;
        curIdx += threadsPerBlock;
    }
}

__global__ void accumulate_kernel_better(int* accumulator, float* edgePixels, float* srcOrient, int* blockStarts, int* blockEnds, int width, int height, int wblock, int hblock, rEntry* entries, int* startPos) {
    int blockIndex = blockIdx.x;
    int start = blockStarts[blockIndex];
    int end = blockEnds[blockIndex];

    int threadIndex = threadIdx.x;
    int index = start + threadIndex;
    if (index > end) return;

    int pixelIndex = static_cast<int>(edgePixels[index]);
    int row = pixelIndex / width;
    int col = pixelIndex % width;

    float phi = srcOrient[pixelIndex]; // gradient direction in [0,360)
    for (int itheta = 0; itheta < cuConstParams.nRotationSlices; itheta++){
        float theta = itheta * cuConstParams.deltaRotationAngle;
        float theta_r = theta / 180.f * cuConstParams.PI;

        // minus mean rotate back by theta
        int iSlice = static_cast<int>(fmodf(phi - theta + 360, 360) / cuConstParams.deltaRotationAngle);
        
        // access RTable and traverse all entries
        int startIdx = startPos[iSlice];
        int endIdx;
        if (iSlice + 1 == cuConstParams.nRotationSlices) endIdx = width * height;
        else endIdx = startPos[iSlice + 1];
        for (int idx = startIdx; idx < endIdx; idx++) {
            rEntry entry = entries[idx];
            float r = entry.r;
            float alpha = entry.alpha;
            for (int is = 0; is < cuConstParams.nScaleSlices; is++){
                float s = is * cuConstParams.deltaScaleRatio + cuConstParams.MINSCALE;
                int xc = col + round(r * s * cos(alpha + theta_r));
                int yc = row + round(r * s * sin(alpha + theta_r));
                if (xc >= 0 && xc < width && yc >= 0 && yc < height) {
                    int accumulatorIndex = is * cuConstParams.nRotationSlices * hblock * wblock
                                           + itheta * hblock * wblock
                                           + yc / cuConstParams.blockSize * wblock
                                           + xc / cuConstParams.blockSize;
                    atomicAdd(accumulator + accumulatorIndex, 1);
                }
            }
        }
    }
}

__global__ void accumulate_kernel_3D(int* accumulator, float* edgePixels, float* srcOrient, int* blockStarts, int* blockEnds, int width, int height, int wblock, int hblock, rEntry* entries, int* startPos) {
    int blockIndex = blockIdx.x;
    int start = blockStarts[blockIndex];
    int end = blockEnds[blockIndex];

    int threadIndex = threadIdx.x;
    int index = start + threadIndex;
    if (index > end) return;

    int itheta = blockIdx.y;
    int is = blockIdx.z;

    int pixelIndex = static_cast<int>(edgePixels[index]);
    int row = pixelIndex / width;
    int col = pixelIndex % width;

    float phi = srcOrient[pixelIndex]; // gradient direction in [0,360)
    float theta = itheta * cuConstParams.deltaRotationAngle;
    float theta_r = theta / 180.f * cuConstParams.PI;

    // minus mean rotate back by theta
    int iSlice = static_cast<int>(fmodf(phi - theta + 360, 360) / cuConstParams.deltaRotationAngle);
    
    float s = is * cuConstParams.deltaScaleRatio + cuConstParams.MINSCALE;
    
    // access RTable and traverse all entries
    int startIdx = startPos[iSlice];
    int endIdx;
    if (iSlice + 1 == cuConstParams.nRotationSlices) endIdx = width * height;
    else endIdx = startPos[iSlice + 1];
    for (int idx = startIdx; idx < endIdx; idx++) {
        rEntry entry = entries[idx];
        float r = entry.r;
        float alpha = entry.alpha;
        int xc = col + round(r * s * cos(alpha + theta_r));
        int yc = row + round(r * s * sin(alpha + theta_r));
        if (xc >= 0 && xc < width && yc >= 0 && yc < height) {
            int accumulatorIndex = is * cuConstParams.nRotationSlices * hblock * wblock
                                   + itheta * hblock * wblock
                                   + yc / cuConstParams.blockSize * wblock
                                   + xc / cuConstParams.blockSize;
            atomicAdd(accumulator + accumulatorIndex, 1);
        }
    }
}

__global__ void accumulate_kernel_3D_sharedMemory(int* accumulator, float* edgePixels, float* srcOrient, int* blockStarts, int* blockEnds, int width, int height, int wblock, int hblock, rEntry* entries, int* startPos) {
    int blockIndex = blockIdx.x;
    int start = blockStarts[blockIndex];
    int end = blockEnds[blockIndex];

    int threadIndex = threadIdx.x;
    int index = start + threadIndex;
    if (index > end) return;

    int itheta = blockIdx.y;
    int is = blockIdx.z;

    int pixelIndex = static_cast<int>(edgePixels[index]);
    int row = pixelIndex / width;
    int col = pixelIndex % width;

    float phi = srcOrient[pixelIndex]; // gradient direction in [0,360)
    float theta = itheta * cuConstParams.deltaRotationAngle;
    float theta_r = theta / 180.f * cuConstParams.PI;

    // minus mean rotate back by theta
    int iSlice = static_cast<int>(fmodf(phi - theta + 360, 360) / cuConstParams.deltaRotationAngle);
    
    float s = is * cuConstParams.deltaScaleRatio + cuConstParams.MINSCALE;

    // dynamically allocate shared memory
    extern __shared__ int accumulatorSlice[];
    
    // access RTable and traverse all entries
    int startIdx = startPos[iSlice];
    int endIdx;
    if (iSlice + 1 == cuConstParams.nRotationSlices) endIdx = width * height;
    else endIdx = startPos[iSlice + 1];
    for (int idx = startIdx; idx < endIdx; idx++) {
        rEntry entry = entries[idx];
        float r = entry.r;
        float alpha = entry.alpha;
        int xc = col + round(r * s * cos(alpha + theta_r));
        int yc = row + round(r * s * sin(alpha + theta_r));
        if (xc >= 0 && xc < width && yc >= 0 && yc < height) {
            // int accumulatorIndex = is * nRotationSlices * hblock * wblock
            //                        + itheta * hblock * wblock
            //                        + yc / blockSize * wblock
            //                        + xc / blockSize;
            // atomicAdd(accumulator + accumulatorIndex, 1);
            int accumulatorSliceIndex = yc / cuConstParams.blockSize * wblock + xc / cuConstParams.blockSize;
            atomicAdd(accumulatorSlice + accumulatorSliceIndex , 1);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        for (int j = 0; j < hblock; j++) {
            for (int i = 0; i < wblock; i++) {
                int accumulatorIndex = is * cuConstParams.nRotationSlices * hblock * wblock
                                       + itheta * hblock * wblock
                                       + j * wblock
                                       + i;
                int accumulatorSliceIndex = j * wblock + i;
                accumulator[accumulatorIndex] += accumulatorSlice[accumulatorSliceIndex];
            }
        }
    }
}

__global__ void findmaxima_kernel(int* accumulator, Point* blockMaxima, int wblock, int hblock) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int row = index / wblock;
    int col = index % wblock;

    int max = -1;
    blockMaxima[index].hits = 0;
    for (int itheta = 0; itheta < cuConstParams.nRotationSlices; itheta++) {
        for (int is = 0; is < cuConstParams.nScaleSlices; is++) {
            int accumulatorIndex = is * cuConstParams.nRotationSlices * hblock * wblock
                                   + itheta * hblock * wblock
                                   + row * wblock
                                   + col;
            if (accumulator[accumulatorIndex] > max) {
                max = accumulator[accumulatorIndex];
                blockMaxima[index].hits = max;
                blockMaxima[index].x = col * cuConstParams.blockSize + cuConstParams.blockSize / 2;
                blockMaxima[index].y = row * cuConstParams.blockSize + cuConstParams.blockSize / 2;
                blockMaxima[index].scale = is * cuConstParams.deltaScaleRatio + cuConstParams.MINSCALE;
                blockMaxima[index].rotation = itheta * cuConstParams.deltaRotationAngle;
            }
        }
    }
}

// srcThreshold, srcOrient: pointers to GPU memory address
void CudaGeneralHoughTransform::accumulate(float* srcThreshold, float* srcOrient, int width, int height, bool naive, int strategy) {
    int wblock = (width + params.blockSize - 1) / params.blockSize;
    int hblock = (height + params.blockSize - 1) / params.blockSize;
    
    // Access using the following pattern:
    // sizeof(accumulator) = arbitrary * sizez * sizey * sizex
    // accumulator[l][k][j][i] = accumulator1D[l*(sizex*sizey*sizez) + k*(sizex*sizey) + j*(sizex) + i]
    int* accumulator;
    Point* blockMaxima;
    Point* hitPointsCuda;

    int sizeAccumulator = params.nScaleSlices * params.nRotationSlices * hblock * wblock;
    cudaMalloc(&accumulator, sizeof(int) * sizeAccumulator);
    cudaMemset(accumulator, 0.f, sizeof(int) * sizeAccumulator);
    cudaMalloc(&blockMaxima, sizeof(Point) * hblock * wblock);

    // Get all edge pixels
    float* edgePixels; // expected data example: [1.0, 3.0, 7.0, ...] (float as it is copied from srcThreshold)
    cudaMalloc(&edgePixels, sizeof(float) * height * width);
    cudaMemset(edgePixels, 0.f, sizeof(int) * height * width);
    thrust::device_ptr<float> srcThresholdThrust = thrust::device_pointer_cast(srcThreshold); 
    thrust::device_ptr<float> edgePixelsThrust = thrust::device_pointer_cast(edgePixels); 
    int numEdgePixels = thrust::copy_if(srcThresholdThrust, srcThresholdThrust + width * height, edgePixelsThrust, is_edge()) - edgePixelsThrust;
    cudaDeviceSynchronize();

    if (naive) {
        // (1) naive approach: 1D partition -> divergent control flow on entries
        // Each block (e.g. 32 threads) take a part of the edge points
        // Each thread take 1 edge point
        // Write to global CUDA memory atomically
        int threadsPerBlock = 32;
        int blocks = (numEdgePixels + threadsPerBlock - 1) / threadsPerBlock;
        accumulate_kernel_naive<<<blocks, threadsPerBlock>>>(accumulator, edgePixels, numEdgePixels, srcOrient, width, height, wblock, hblock, entries, startPos);
        cudaDeviceSynchronize();
    } else {
        // (2) better approach: 
        //     a. put edge points into buckets by phi
        //     b. points with the same phi go together in a kernel (1D) each block has the same phi value
        
        // sort edgePixels by angle
        thrust::sort(edgePixelsThrust, edgePixelsThrust + numEdgePixels, compare_pixel_by_orient(srcOrient));
        cudaDeviceSynchronize();
        
        // traverse sorted edge pixels and find the seperation points O(N)
        // Possible optimization: binary search O(logN) to find all the seperation points (72), can parallel the bineary search
        // Use 1 kernel to calculate the following:
        // GPU memory: int[nRotationSlices] (start index for each slice)
        // CPU memory: number of blocks (calculate using threadsPerBlock)
        int* startIndex;
        cudaMalloc(&startIndex, sizeof(int) * params.nRotationSlices);
        int* numBlocksDevice;
        cudaMalloc(&numBlocksDevice, sizeof(int));
        int threadsPerBlock = 32;
        findCutoffs_kernel<<<1, 1>>>(edgePixels, numEdgePixels, srcOrient, threadsPerBlock, startIndex, numBlocksDevice);
        cudaDeviceSynchronize();

        int numBlocks;
        cudaMemcpy(&numBlocks, numBlocksDevice, sizeof(int), cudaMemcpyDeviceToHost); 
        cudaFree(numBlocksDevice);

        // allocate int[num of blocks] x2 for start and end index of each block
        int* blockStarts;
        int* blockEnds;
        cudaMalloc(&blockStarts, sizeof(int) * numBlocks);
        cudaMalloc(&blockEnds, sizeof(int) * numBlocks);
        fillIntervals_kernel<<<1, 1>>>(startIndex, numEdgePixels, threadsPerBlock, blockStarts, blockEnds);
        cudaDeviceSynchronize();

        switch (strategy) {
            case 0:
                {
                    accumulate_kernel_better<<<numBlocks, threadsPerBlock>>>(accumulator, edgePixels, srcOrient, blockStarts, blockEnds, width, height, wblock, hblock, entries, startPos);
                    cudaDeviceSynchronize();
                }
                break;
            case 1:
                {
                    // (3) more parallelism: 
                    //     a. put edge points into buckets by phi
                    //     b. go by same phi, then same theta, then same scale (3D kernel)
                    dim3 gridDim(numBlocks, params.nRotationSlices, params.nScaleSlices);
                    accumulate_kernel_3D<<<gridDim, threadsPerBlock>>>(accumulator, edgePixels, srcOrient, blockStarts, blockEnds, width, height, wblock, hblock, entries, startPos);
                    cudaDeviceSynchronize();
                }
                break;
            case 2:
                {
                    // shared memory of accumulator slice
                    dim3 gridDim(numBlocks, params.nRotationSlices, params.nScaleSlices);
                    int sharedSize = hblock * wblock * sizeof(int);
                    accumulate_kernel_3D_sharedMemory<<<gridDim, threadsPerBlock, sharedSize>>>(accumulator, edgePixels, srcOrient, blockStarts, blockEnds, width, height, wblock, hblock, entries, startPos);
                    cudaDeviceSynchronize();
                }
                break;
            default:
                printf("Please select strategy 0, 1, 2");
        }

        cudaFree(startIndex);
        cudaFree(blockStarts);
        cudaFree(blockEnds);
    }

    // Use a seperate kernel to fill in the blockMaxima array (avoid extra memory read)
    int threadsPerBlock = 32;
    int blocks = (hblock * wblock + threadsPerBlock - 1) / threadsPerBlock;
    findmaxima_kernel<<<blocks, threadsPerBlock>>>(accumulator, blockMaxima, wblock, hblock);
    cudaDeviceSynchronize();

    // use thrust::max_element to get the max hit from blockMaxima
    thrust::device_ptr<Point> blockMaximaThrust = thrust::device_pointer_cast(blockMaxima);
    Point maxPoint = *(thrust::max_element(blockMaximaThrust, blockMaximaThrust + hblock * wblock, compare_point()));
    cudaDeviceSynchronize();
    int maxHit = maxPoint.hits;

    // use thrust::copy_if to get the above threshold points & number
    cudaMalloc(&hitPointsCuda, sizeof(Point) * numEdgePixels);
    thrust::device_ptr<Point> hitPointsThrust = thrust::device_pointer_cast(hitPointsCuda);
    int numResPoints = thrust::copy_if(blockMaximaThrust, blockMaximaThrust + hblock * wblock, hitPointsThrust, filter_point(round(maxHit * params.thresRatio))) - blockMaximaThrust;
    cudaDeviceSynchronize();

    // Copy back to cpu memory
    hitPoints.clear();
    hitPoints.resize(numResPoints);
    cudaMemcpy(&hitPoints[0], hitPointsCuda, numResPoints * sizeof(Point), cudaMemcpyDeviceToHost);

    cudaFree(accumulator);
    cudaFree(blockMaxima);
    cudaFree(edgePixels);
    cudaFree(hitPointsCuda);
}

// 1. no need to allocate each image in cpu
// 2. convolve/magnitude/orientation/nms/threshold/createRTable: each thread -> pixel
// 3. parallel convolveX, convolveY
// 4. parallel magnitude & orientation
void CudaGeneralHoughTransform::processTemplate() {
    printf("----------Start processing template----------\n");
    unsigned char* deviceTplData;
    float* tplGrayData;
    float* mag;
    float* orient;

    cudaMalloc(&deviceTplData, 3 * tpl->width * tpl->height * sizeof(unsigned char));
    cudaMemcpy(deviceTplData, tpl->data, 3 * tpl->width * tpl->height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMalloc(&tplGrayData, tpl->width * tpl->height * sizeof(float));

    cudaMalloc(&entries, tpl->width * tpl->height * sizeof(rEntry));
    cudaMalloc(&mag, tpl->width * tpl->height * sizeof(float));
    cudaMalloc(&orient, tpl->width * tpl->height * sizeof(float));

    dim3 blockDim(TPB_X, TPB_Y, 1);
    dim3 gridDim((tpl->width + TPB_X - 1) / TPB_X, 
                 (tpl->height + TPB_Y - 1) / TPB_Y, 1);
    kernelConvertToGray<<<gridDim, blockDim>>>(deviceTplData, tplGrayData, tpl->width, tpl->height);
    cudaDeviceSynchronize();
    // GrayImage* grayTpl = new GrayImage;
    // grayTpl->setGrayImage(tpl->width, tpl->height);
    // cudaMemcpy(grayTpl->data, tplGrayData, tpl->width * tpl->height * sizeof(float), cudaMemcpyDeviceToHost);
    // writeGrayPPMImage(grayTpl, "gray1.ppm");

    kernelProcessStep1<<<gridDim, blockDim>>>(tplGrayData, mag, orient, tpl->width, tpl->height);
    cudaDeviceSynchronize();

    GrayImage* magTpl = new GrayImage;
    // magTpl->setGrayImage(tpl->width, tpl->height);
    // cudaMemcpy(magTpl->data, mag, tpl->width * tpl->height * sizeof(float), cudaMemcpyDeviceToHost);
    // writeGrayPPMImage(magTpl, "mag1.ppm");
    // GrayImage* orientTpl = new GrayImage;
    // orientTpl->setGrayImage(tpl->width, tpl->height);
    // cudaMemcpy(orientTpl->data, orient, tpl->width * tpl->height * sizeof(float), cudaMemcpyDeviceToHost);
    // writeGrayPPMImage(orientTpl, "orient1.ppm");

    kernelProcessStep2<<<gridDim, blockDim>>>(mag, orient, entries, tpl->width, tpl->height, true);
    cudaDeviceSynchronize();  

    // cudaMemcpy(magTpl->data, mag, tpl->width * tpl->height * sizeof(float), cudaMemcpyDeviceToHost);
    // writeGrayPPMImage(magTpl, "threshold1.ppm");

    
    thrust::device_ptr<rEntry> entriesThrust = thrust::device_pointer_cast(entries); 

    thrust::sort(entriesThrust, entriesThrust + tpl->width * tpl->height, compare_entry_by_rotation());
    cudaDeviceSynchronize(); 

    cudaMalloc(&startPos, params.nRotationSlices * sizeof(int));
    processRTable<<<1, 1>>>(entries, startPos, tpl->width, tpl->height);
    
    // memory deallocation
    cudaCheckError(cudaFree(deviceTplData));
    cudaCheckError(cudaFree(orient));
    cudaCheckError(cudaFree(tplGrayData));
    cudaCheckError(cudaFree(mag));
    printf("----------End Processing Template----------\n");
}

void CudaGeneralHoughTransform::accumulateSource() {
    unsigned char* deviceSrcData;
    float* srcGrayData;
    float* orient;
    float* mag;

    cudaMalloc(&deviceSrcData, 3 * src->width * src->height * sizeof(unsigned char));
    cudaMalloc(&srcGrayData, src->width * src->height * sizeof(float));
    cudaMemcpy(deviceSrcData, src->data, 3 * src->width * src->height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    //cudaMemcpyToSymbol(grayData, &srcGrayData, sizeof(float*));
    cudaMalloc(&mag, tpl->width * tpl->height * sizeof(float));
    cudaMalloc(&orient, src->width * src->height * sizeof(float));
    //cudaMemcpyToSymbol(orient, &tmpOrientation, sizeof(float*));

    dim3 blockDim(TPB_X, TPB_Y, 1);
    dim3 gridDim((src->width + TPB_X - 1) / TPB_X, 
                 (src->height + TPB_Y - 1) / TPB_Y, 1);
    
    kernelConvertToGray<<<gridDim, blockDim>>>(deviceSrcData, srcGrayData, src->width, src->height);
    cudaDeviceSynchronize();

    kernelProcessStep1<<<gridDim, blockDim>>>(srcGrayData, mag, orient, src->width, src->height);
    cudaDeviceSynchronize();

    kernelProcessStep2<<<gridDim, blockDim>>>(mag, orient, entries, src->width, src->height, false);
    cudaDeviceSynchronize();  

    // GrayImage* magThreshold = new GrayImage;
    // cudaMemcpyFromSymbol(magThreshold->data, grayData, src->width * src->height * sizeof(float), cudaMemcpyDeviceToHost);

    double startAccumulateTime = CycleTimer::currentSeconds();
    accumulate(mag, orient, src->width, src->height, true, -1);
    double endAccumulateTime = CycleTimer::currentSeconds();
    double totalAccumulateTime = endAccumulateTime - startAccumulateTime;
    printf("Accumulate:        %.4f ms\n", 1000.f * totalAccumulateTime);
    
    // memory deallocation
    cudaFree(srcGrayData);
    cudaFree(orient);
    cudaFree(deviceSrcData);
    cudaFree(mag);
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
