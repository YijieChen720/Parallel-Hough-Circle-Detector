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
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

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
bool keepPixel(const float* magnitude, int i, int j, int width, int height, int gradient) {
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
    
    float neighbourOne, neighbourTwo;
    if (neighbourOnei < 0 || neighbourOnei >= width || neighbourOnej < 0 || neighbourOnej >= height) neighbourOne = 0.f;
    else neighbourOne = magnitude[neighbourOnej * width + neighbourOnei];
    if (neighbourTwoi < 0 || neighbourTwoi >= width || neighbourTwoj < 0 || neighbourTwoj >= height) neighbourTwo = 0.f;
    else neighbourTwo = magnitude[neighbourTwoj * width + neighbourTwoi];
    float cur = magnitude[j * width + i];
    
    return (neighbourOne <= cur) && (neighbourTwo <= cur);
}

__device__ __inline__
void edgenms(float* mag, float* orientation, float* result, int width, int height, int i, int j, int localIdx) {
    int pixelGradient = static_cast<int>(orientation[localIdx] / 45) * 45 % 180;
    if (keepPixel(mag, i, j, width, height, pixelGradient)) {
        result[localIdx] = mag[j * width + i];
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

    __shared__ float orientation[TPB]; 
    __shared__ float magThreshold[TPB];
    orientation[localIdx] = orient[index]; 
    __syncthreads();
    
    edgenms(mag, orientation, magThreshold, width, height, indexX, indexY, localIdx); 
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


struct is_edge
{
    __host__ __device__
    bool operator() (const float x)
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

__global__ void accumulate_kernel_naive(int* accumulator, float* edgePixels, int numEdgePixels, float* srcOrient, int width, int height, int wblock, int hblock, rEntry* entries, int* startPos, int tplSize) {
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
        if (iSlice + 1 == cuConstParams.nRotationSlices) endIdx = tplSize;
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

__global__ void accumulate_kernel_naive_3D(int* accumulator, float* edgePixels, int numEdgePixels, float* srcOrient, int width, int height, int wblock, int hblock, rEntry* entries, int* startPos, int tplSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numEdgePixels) return;

    int pixelIndex = static_cast<int>(edgePixels[index]);
    int row = pixelIndex / width;
    int col = pixelIndex % width;
    
    int itheta = blockIdx.y;
    int is = blockIdx.z;

    float phi = srcOrient[pixelIndex]; // gradient direction in [0,360)
    float theta = itheta * cuConstParams.deltaRotationAngle;
    float theta_r = theta / 180.f * cuConstParams.PI;

    // minus mean rotate back by theta
    int iSlice = static_cast<int>(fmodf(phi - theta + 360, 360) / cuConstParams.deltaRotationAngle);
    
    float s = is * cuConstParams.deltaScaleRatio + cuConstParams.MINSCALE;

    // access RTable and traverse all entries
    int startIdx = startPos[iSlice];
    int endIdx;
    if (iSlice + 1 == cuConstParams.nRotationSlices) endIdx = tplSize;
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

__global__ void accumulate_kernel_binning (int* accumulator, float* edgePixels, float* srcOrient, int* blockStarts, int* blockEnds, int width, int height, int wblock, int hblock, rEntry* entries, int* startPos, int tplSize) {
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
        if (iSlice + 1 == cuConstParams.nRotationSlices) endIdx = tplSize;
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

__global__ void accumulate_kernel_binning_3D(int* accumulator, float* edgePixels, float* srcOrient, int* blockStarts, int* blockEnds, int width, int height, int wblock, int hblock, rEntry* entries, int* startPos, int tplSize) {
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
    if (iSlice + 1 == cuConstParams.nRotationSlices) endIdx = tplSize;
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

__global__ void accumulate_kernel_binning_3D_sharedMemory(int* accumulator, float* edgePixels, float* srcOrient, int* blockStarts, int* blockEnds, int width, int height, int wblock, int hblock, rEntry* entries, int* startPos, int tplSize) {
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
    int threadCnt = end - start + 1;
    int coverage = hblock * wblock / threadCnt;
    int sStart = coverage * threadIdx.x;
    int eEnd = sStart + coverage; 
    for (int i = sStart; i < eEnd; i++) accumulatorSlice[i] = 0;
    if (threadIdx.x == 0) {
        for (int i = coverage * threadCnt; i < wblock * hblock; i++) accumulatorSlice[i] = 0;
    }

    __syncthreads();

    // access RTable and traverse all entries
    int startIdx = startPos[iSlice];
    int endIdx;
    if (iSlice + 1 == cuConstParams.nRotationSlices) endIdx = tplSize;
    else endIdx = startPos[iSlice + 1];
    for (int idx = startIdx; idx < endIdx; idx++) {
        rEntry entry = entries[idx];
        float r = entry.r;
        float alpha = entry.alpha;
        int xc = col + round(r * s * cos(alpha + theta_r));
        int yc = row + round(r * s * sin(alpha + theta_r));
        if (xc >= 0 && xc < width && yc >= 0 && yc < height) {
            int accumulatorSliceIndex = yc / cuConstParams.blockSize * wblock + xc / cuConstParams.blockSize;
            atomicAdd(accumulatorSlice + accumulatorSliceIndex, 1);
        }
    }

    __syncthreads();
    for (int i = sStart; i < eEnd; i++) {
        int accumulatorIndex = is * cuConstParams.nRotationSlices * hblock * wblock
                               + itheta * hblock * wblock
                               + i;
        int accumulatorSliceIndex = i;
        atomicAdd(accumulator + accumulatorIndex, accumulatorSlice[accumulatorSliceIndex]);
    }
    if (threadIdx.x == 0) {
        for (int i = coverage * threadCnt; i < wblock * hblock; i++) {
            int accumulatorIndex = is * cuConstParams.nRotationSlices * hblock * wblock
                                   + itheta * hblock * wblock
                                   + i;
            int accumulatorSliceIndex = i;
            atomicAdd(accumulator + accumulatorIndex, accumulatorSlice[accumulatorSliceIndex]);
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
void CudaGeneralHoughTransform::accumulate(float* srcThreshold, float* srcOrient, int width, int height, bool naive, bool sort, int strategy) {
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
    cudaCheckError(cudaDeviceSynchronize());

    if (naive) {
        // naive approach: 1D partition -> divergent control flow on entries
        // Each block (e.g. 32 threads) take a part of the edge points
        // Write to global CUDA memory atomically
        int threadsPerBlock = 32;
        int blocks = (numEdgePixels + threadsPerBlock - 1) / threadsPerBlock;

        switch (strategy) {
            case 0: 
                {
                    accumulate_kernel_naive<<<blocks, threadsPerBlock>>>(accumulator, edgePixels, numEdgePixels, srcOrient, width, height, wblock, hblock, entries, startPos, tpl->width * tpl->height);
                    cudaCheckError(cudaDeviceSynchronize());
                }
                break;
            case 1:
                {
                    dim3 gridDim(blocks, params.nRotationSlices, params.nScaleSlices);
                    accumulate_kernel_naive_3D<<<gridDim, threadsPerBlock>>>(accumulator, edgePixels, numEdgePixels, srcOrient, width, height, wblock, hblock, entries, startPos, tpl->width * tpl->height);
                    cudaCheckError(cudaDeviceSynchronize());
                }
                break;
            default:
                printf("Please select strategy 0, 1\n");
        }
    } else if (sort) {
        int threadsPerBlock = 32;
        int blocks = (numEdgePixels + threadsPerBlock - 1) / threadsPerBlock;
        thrust::sort(edgePixelsThrust, edgePixelsThrust + numEdgePixels, compare_pixel_by_orient(srcOrient));
        cudaCheckError(cudaDeviceSynchronize());
        switch (strategy) {
            case 0: 
                {
                    accumulate_kernel_naive<<<blocks, threadsPerBlock>>>(accumulator, edgePixels, numEdgePixels, srcOrient, width, height, wblock, hblock, entries, startPos, tpl->width * tpl->height);
                    cudaCheckError(cudaDeviceSynchronize());
                }
                break;
            case 1:
                {
                    double startKernelTime = CycleTimer::currentSeconds();
                    dim3 gridDim(blocks, params.nRotationSlices, params.nScaleSlices);
                    accumulate_kernel_naive_3D<<<gridDim, threadsPerBlock>>>(accumulator, edgePixels, numEdgePixels, srcOrient, width, height, wblock, hblock, entries, startPos, tpl->width * tpl->height);
                    cudaCheckError(cudaDeviceSynchronize());
                    double endKernelTime = CycleTimer::currentSeconds();
                    double kernelTime = endKernelTime - startKernelTime;
                    printf("Kernel:            %.4f ms\n", 1000.f * kernelTime);
                }
                break;
            default:
                printf("Please select strategy 0, 1\n");
        }
    } else {
        // sort edgePixels by angle
        thrust::sort(edgePixelsThrust, edgePixelsThrust + numEdgePixels, compare_pixel_by_orient(srcOrient));
        cudaCheckError(cudaDeviceSynchronize());
        
        double startBinningTime = CycleTimer::currentSeconds();
        // GPU memory: int[nRotationSlices] (start index for each slice)
        // CPU memory: number of blocks (calculate using threadsPerBlock)
        int* startIndex;
        cudaMalloc(&startIndex, sizeof(int) * params.nRotationSlices);
        int* numBlocksDevice;
        cudaMalloc(&numBlocksDevice, sizeof(int));
        int threadsPerBlock = 32;
        findCutoffs_kernel<<<1, 1>>>(edgePixels, numEdgePixels, srcOrient, threadsPerBlock, startIndex, numBlocksDevice);
        cudaCheckError(cudaDeviceSynchronize());

        int numBlocks;
        cudaMemcpy(&numBlocks, numBlocksDevice, sizeof(int), cudaMemcpyDeviceToHost); 
        cudaFree(numBlocksDevice);
        printf("num blocks: %d\n", numBlocks);

        // allocate int[num of blocks] x2 for start and end index of each block
        int* blockStarts;
        int* blockEnds;
        cudaMalloc(&blockStarts, sizeof(int) * numBlocks);
        cudaMalloc(&blockEnds, sizeof(int) * numBlocks);
        fillIntervals_kernel<<<1, 1>>>(startIndex, numEdgePixels, threadsPerBlock, blockStarts, blockEnds);
        cudaCheckError(cudaDeviceSynchronize());
        double endBinningTime = CycleTimer::currentSeconds();
        double binningTime = endBinningTime - startBinningTime;
        printf("Binning:           %.4f ms\n", 1000.f * binningTime);

        switch (strategy) {
            case 0:
                {
                    accumulate_kernel_binning<<<numBlocks, threadsPerBlock>>>(accumulator, edgePixels, srcOrient, blockStarts, blockEnds, width, height, wblock, hblock, entries, startPos, tpl->width * tpl->height);
                    cudaCheckError(cudaDeviceSynchronize());
                }
                break;
            case 1:
                {
                    double startKernelTime = CycleTimer::currentSeconds();
                    dim3 gridDim(numBlocks, params.nRotationSlices, params.nScaleSlices);
                    accumulate_kernel_binning_3D<<<gridDim, threadsPerBlock>>>(accumulator, edgePixels, srcOrient, blockStarts, blockEnds, width, height, wblock, hblock, entries, startPos, tpl->width * tpl->height);
                    cudaCheckError(cudaDeviceSynchronize());
                    double endKernelTime = CycleTimer::currentSeconds();
                    double kernelTime = endKernelTime - startKernelTime;
                    printf("Kernel:            %.4f ms\n", 1000.f * kernelTime);
                }
                break;
            case 2:
                {
                    double startKernelTime = CycleTimer::currentSeconds();
                    // shared memory of accumulator slice
                    dim3 gridDim(numBlocks, params.nRotationSlices, params.nScaleSlices);
                    int sharedSize = hblock * wblock * sizeof(int);
                    accumulate_kernel_binning_3D_sharedMemory<<<gridDim, threadsPerBlock, sharedSize>>>(accumulator, edgePixels, srcOrient, blockStarts, blockEnds, width, height, wblock, hblock, entries, startPos, tpl->width * tpl->height);
                    cudaCheckError(cudaDeviceSynchronize());
                    double endKernelTime = CycleTimer::currentSeconds();
                    double kernelTime = endKernelTime - startKernelTime;
                    printf("Kernel:            %.4f ms\n", 1000.f * kernelTime);
                }
                break;
            default:
                printf("Please select strategy 0, 1, 2\n");
        }

        cudaFree(startIndex);
        cudaFree(blockStarts);
        cudaFree(blockEnds);
    }

    // Use a seperate kernel to fill in the blockMaxima array (avoid extra memory read)
    int threadsPerBlock = 32;
    int blocks = (hblock * wblock + threadsPerBlock - 1) / threadsPerBlock;
    findmaxima_kernel<<<blocks, threadsPerBlock>>>(accumulator, blockMaxima, wblock, hblock);
    cudaCheckError(cudaDeviceSynchronize());

    // use thrust::max_element to get the max hit from blockMaxima
    thrust::device_ptr<Point> blockMaximaThrust = thrust::device_pointer_cast(blockMaxima);
    Point maxPoint = *(thrust::max_element(blockMaximaThrust, blockMaximaThrust + hblock * wblock, compare_point()));
    cudaCheckError(cudaDeviceSynchronize());
    int maxHit = maxPoint.hits;

    // use thrust::copy_if to get the above threshold points & number
    cudaMalloc(&hitPointsCuda, sizeof(Point) * numEdgePixels);
    thrust::device_ptr<Point> hitPointsThrust = thrust::device_pointer_cast(hitPointsCuda);
    int numResPoints = thrust::copy_if(blockMaximaThrust, blockMaximaThrust + hblock * wblock, hitPointsThrust, filter_point(round(maxHit * params.thresRatio))) - hitPointsThrust;
    cudaCheckError(cudaDeviceSynchronize());

    // Copy back to cpu memory
    hitPoints.clear();
    hitPoints.resize(numResPoints);
    cudaMemcpy(&hitPoints[0], hitPointsCuda, numResPoints * sizeof(Point), cudaMemcpyDeviceToHost);

    cudaFree(accumulator);
    cudaFree(blockMaxima);
    cudaFree(edgePixels);
    cudaFree(hitPointsCuda);

    for (auto p : hitPoints) {
        printf("hit points: %d %d hits: %d scale: %f rotation: %f\n", p.x, p.y, p.hits, p.scale, p.rotation);
    }
}

void CudaGeneralHoughTransform::processTemplate() {
    printf("----------Start processing template----------\n");
    double startAllocateTime = CycleTimer::currentSeconds();
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
    double endAllocateTime = CycleTimer::currentSeconds();

    dim3 blockDim(TPB_X, TPB_Y, 1);
    dim3 gridDim((tpl->width + TPB_X - 1) / TPB_X, 
                 (tpl->height + TPB_Y - 1) / TPB_Y, 1);
    double startGrayTime = CycleTimer::currentSeconds();
    kernelConvertToGray<<<gridDim, blockDim>>>(deviceTplData, tplGrayData, tpl->width, tpl->height);
    cudaDeviceSynchronize();
    double endGrayTime = CycleTimer::currentSeconds();

    double startStep1Time = CycleTimer::currentSeconds();
    kernelProcessStep1<<<gridDim, blockDim>>>(tplGrayData, mag, orient, tpl->width, tpl->height);
    cudaDeviceSynchronize();
    double endStep1Time = CycleTimer::currentSeconds();

    double startStep2Time = CycleTimer::currentSeconds();
    kernelProcessStep2<<<gridDim, blockDim>>>(mag, orient, entries, tpl->width, tpl->height, true);
    cudaDeviceSynchronize();  
    double endStep2Time = CycleTimer::currentSeconds();

    double startSortTime = CycleTimer::currentSeconds();
    thrust::device_ptr<rEntry> entriesThrust = thrust::device_pointer_cast(entries); 
    thrust::sort(entriesThrust, entriesThrust + tpl->width * tpl->height, compare_entry_by_rotation());
    double endSortTime = CycleTimer::currentSeconds();

    double startRTableTime = CycleTimer::currentSeconds();
    cudaMalloc(&startPos, params.nRotationSlices * sizeof(int));
    thrust::device_vector<rEntry> rTableThrust(entriesThrust, entriesThrust + tpl->width * tpl->height);
    thrust::device_vector<rEntry> values(params.nRotationSlices);
    for (int i = 0; i < params.nRotationSlices; i++){
        values[i] = (struct rEntry){i, 0.f, 0.f};
    }
    thrust::device_vector<int> posThrust(params.nRotationSlices);
    thrust::lower_bound(rTableThrust.begin(), rTableThrust.end(), values.begin(), values.end(), posThrust.begin(), compare_entry_by_rotation());

    int* posPtr = thrust::raw_pointer_cast(&posThrust[0]);
    cudaMemcpy(startPos, posPtr, params.nRotationSlices * sizeof(int), cudaMemcpyHostToDevice);

    double endRTableTime = CycleTimer::currentSeconds();

    // memory deallocation
    double startFreeTime = CycleTimer::currentSeconds();
    cudaFree(deviceTplData);
    cudaFree(orient);
    cudaFree(tplGrayData);
    cudaFree(mag);
    double endFreeTime = CycleTimer::currentSeconds();

    double allocateTime = endAllocateTime - startAllocateTime;
    double grayTime = endGrayTime - startGrayTime;
    double step1Time = endStep1Time - startStep1Time;
    double step2Time = endStep2Time - startStep2Time;
    double sortTime = endSortTime - startSortTime;
    double RTableTime = endRTableTime - startRTableTime;
    double freeTime = endFreeTime - startFreeTime;
    printf("Allocate memory:   %.4f ms\n", 1000.f * allocateTime);
    printf("Convert to Gray:   %.4f ms\n", 1000.f * grayTime);
    printf("Step1:             %.4f ms\n", 1000.f * step1Time);
    printf("Step2:             %.4f ms\n", 1000.f * step2Time);
    printf("Sort:              %.4f ms\n", 1000.f * sortTime);
    printf("Process R table:   %.4f ms\n", 1000.f * RTableTime);
    printf("Free Memory:       %.4f ms\n", 1000.f * freeTime);
    
    printf("----------End Processing Template----------\n");
}

void CudaGeneralHoughTransform::accumulateSource(bool naive, bool sort, bool is1D) {
    printf("----------Start processing and accumulating source----------\n");
    double startAllocateTime = CycleTimer::currentSeconds();
    unsigned char* deviceSrcData;
    float* srcGrayData;
    float* orient;
    float* mag;

    cudaMalloc(&deviceSrcData, 3 * src->width * src->height * sizeof(unsigned char));
    cudaMalloc(&srcGrayData, src->width * src->height * sizeof(float));
    cudaMemcpy(deviceSrcData, src->data, 3 * src->width * src->height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMalloc(&mag, src->width * src->height * sizeof(float));
    cudaMalloc(&orient, src->width * src->height * sizeof(float));

    double endAllocateTime = CycleTimer::currentSeconds();

    dim3 blockDim(TPB_X, TPB_Y, 1);
    dim3 gridDim((src->width + TPB_X - 1) / TPB_X, 
                 (src->height + TPB_Y - 1) / TPB_Y, 1);
    
    double startGrayTime = CycleTimer::currentSeconds();
    kernelConvertToGray<<<gridDim, blockDim>>>(deviceSrcData, srcGrayData, src->width, src->height);
    cudaDeviceSynchronize();
    double endGrayTime = CycleTimer::currentSeconds();

    double startStep1Time = CycleTimer::currentSeconds();
    kernelProcessStep1<<<gridDim, blockDim>>>(srcGrayData, mag, orient, src->width, src->height);
    cudaDeviceSynchronize();
    double endStep1Time = CycleTimer::currentSeconds();

    double startStep2Time = CycleTimer::currentSeconds();
    kernelProcessStep2<<<gridDim, blockDim>>>(mag, orient, entries, src->width, src->height, false);
    cudaDeviceSynchronize();  
    double endStep2Time = CycleTimer::currentSeconds();

    double startAccumulateTime = CycleTimer::currentSeconds();
    int dimension = is1D ? 0 : 1;
    accumulate(mag, orient, src->width, src->height, naive, sort, dimension);
    double endAccumulateTime = CycleTimer::currentSeconds();
    double allocateTime = endAllocateTime - startAllocateTime;
    double grayTime = endGrayTime - startGrayTime;
    double step1Time = endStep1Time - startStep1Time;
    double step2Time = endStep2Time - startStep2Time;
    double totalAccumulateTime = endAccumulateTime - startAccumulateTime;
    printf("Allocate memory:   %.4f ms\n", 1000.f * allocateTime);
    printf("Convert to Gray:   %.4f ms\n", 1000.f * grayTime);
    printf("Step1:             %.4f ms\n", 1000.f * step1Time);
    printf("Step2:             %.4f ms\n", 1000.f * step2Time);
    printf("Accumulate:        %.4f ms\n", 1000.f * totalAccumulateTime);
    
    // memory deallocation
    double startFreeTime = CycleTimer::currentSeconds();
    cudaFree(srcGrayData);
    cudaFree(orient);
    cudaFree(deviceSrcData);
    cudaFree(mag);
    double endFreeTime = CycleTimer::currentSeconds();
    double freeTime = endFreeTime - startFreeTime;
    printf("Free Memory:       %.4f ms\n", 1000.f * freeTime);
    printf("----------End processing and accumulating source----------\n");
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
