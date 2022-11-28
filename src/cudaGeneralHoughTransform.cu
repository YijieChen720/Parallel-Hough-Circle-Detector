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

struct GlobalConstants {
    int width;
    int height;
}

__constant__ GlobalConstants cuTemplateParams;
__constant__ GlobalConstants cuSourceParams;

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
    bool operator()(Point &p1, Point &p2)
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
        return orient[idx1] < orient[idx2];
    }
};

__global__ void accumulate_kernel_naive(int* accumulator, float* edgePixels, int numEdgePixels, float* srcOrient) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numEdgePixels) return;

    int pixelIndex = static_cast<int>(edgePixels[index]);
    float phi = srcOrient[pixelIndex]; // gradient direction in [0,360)
    for (int itheta = 0; itheta < nRotationSlices; itheta++){
        float theta = itheta * deltaRotationAngle;
        float theta_r = theta / 180.f * PI;

        // minus mean rotate back by theta
        int iSlice = static_cast<int>(fmod(phi - theta + 360, 360) / deltaRotationAngle);
        
        // access RTable and traverse all entries
        // std::vector<rEntry> entries = rTable[iSlice];
        for (auto entry: entries){
            float r = entry.r;
            float alpha = entry.alpha;
            for (int is = 0; is < nScaleSlices; is++){
                float s = is * deltaScaleRatio + MINSCALE;
                int xc = i + round(r * s * cos(alpha + theta_r));
                int yc = j + round(r * s * sin(alpha + theta_r));
                if (xc >= 0 && xc < cuSourceParams.width && yc >= 0 && yc < cuSourceParams.height) {
                    int accumulatorIndex = is * nRotationSlices * (cuSourceParams.height / blockSize + 1) * (cuSourceParams.width / blockSize + 1) 
                                           + itheta * (cuSourceParams.height / blockSize + 1) * (cuSourceParams.width / blockSize + 1)
                                           + yc / blockSize * (cuSourceParams.width / blockSize + 1)
                                           + xc / blockSize;
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
        int iSlice = static_cast<int>(phi / deltaRotationAngle);
        if (iSlice > cur) {
            for (int j = cur + 1; j <= iSlice; j++) {
                startIndex[j] = i;
            }
            *numBlocksDevice += (startIndex[cur + 1] - startIndex[cur] + threadsPerBlock - 1) / threadsPerBlock;
            cur = iSlice;
        }
    }
    *numBlocksDevice += (numEdgePixels - startIndex[cur] + threadsPerBlock - 1) / threadsPerBlock;
    if (cur < nRotationSlices - 1) {
        for (int i = cur + 1; i < nRotationSlices; i++) {
            startIndex[i] = numEdgePixels;
        }
    }
}

__global__ void fillIntervals_kernel(int* startIndex, int numEdgePixels, int threadsPerBlock, int* blockStarts, int* blockEnds) {
    int cur = 0;
    int idx = 0;
    int curIdx = startIndex[cur];
    while (cur < nRotationSlices - 1) {
        if (curIdx + threadsPerBlock < startIndex[cur + 1]) {
            blockStarts[idx] = curIdx;
            blockEnds[idx] = curIdx + threadsPerBlock - 1;
            curIdx += threadsPerBlock;
            idx++;
        } else if (startIndex[cur] != startIndex[cur + 1])
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

__global__ void accumulate_kernel_better(int* accumulator, float* edgePixels, float* srcOrient, int* blockStarts, int* blockEnds) {
    int blockIndex = blockIdx.x;
    int start = blockStarts[blockIdx];
    int end = blockEnds[blockIdx];

    int threadIndex = threadIdx.x;
    int index = start + threadIndex;
    if (index > end) return;

    int pixelIndex = static_cast<int>(edgePixels[index]);
    float phi = srcOrient[pixelIndex]; // gradient direction in [0,360)
    for (int itheta = 0; itheta < nRotationSlices; itheta++){
        float theta = itheta * deltaRotationAngle;
        float theta_r = theta / 180.f * PI;

        // minus mean rotate back by theta
        int iSlice = static_cast<int>(fmod(phi - theta + 360, 360) / deltaRotationAngle);
        
        // access RTable and traverse all entries
        // std::vector<rEntry> entries = rTable[iSlice];
        for (auto entry: entries){
            float r = entry.r;
            float alpha = entry.alpha;
            for (int is = 0; is < nScaleSlices; is++){
                float s = is * deltaScaleRatio + MINSCALE;
                int xc = i + round(r * s * cos(alpha + theta_r));
                int yc = j + round(r * s * sin(alpha + theta_r));
                if (xc >= 0 && xc < cuSourceParams.width && yc >= 0 && yc < cuSourceParams.height) {
                    int accumulatorIndex = is * nRotationSlices * (cuSourceParams.height / blockSize + 1) * (cuSourceParams.width / blockSize + 1) 
                                           + itheta * (cuSourceParams.height / blockSize + 1) * (cuSourceParams.width / blockSize + 1)
                                           + yc / blockSize * (cuSourceParams.width / blockSize + 1)
                                           + xc / blockSize;
                    atomicAdd(accumulator + accumulatorIndex, 1);
                }
            }
        }
    }
}

__global__ void accumulate_kernel_3D(int* accumulator, float* edgePixels, float* srcOrient, int* blockStarts, int* blockEnds) {
    int blockIndex = blockIdx.x;
    int start = blockStarts[blockIdx];
    int end = blockEnds[blockIdx];

    int threadIndex = threadIdx.x;
    int index = start + threadIndex;
    if (index > end) return;

    int itheta = blockIdx.y;
    int is = blockIdx.z;

    int pixelIndex = static_cast<int>(edgePixels[index]);
    float phi = srcOrient[pixelIndex]; // gradient direction in [0,360)
    float theta = itheta * deltaRotationAngle;
    float theta_r = theta / 180.f * PI;

    // minus mean rotate back by theta
    int iSlice = static_cast<int>(fmod(phi - theta + 360, 360) / deltaRotationAngle);
    
    float s = is * deltaScaleRatio + MINSCALE;
    
    // access RTable and traverse all entries
    // std::vector<rEntry> entries = rTable[iSlice];
    for (auto entry: entries){
        float r = entry.r;
        float alpha = entry.alpha;
        int xc = i + round(r * s * cos(alpha + theta_r));
        int yc = j + round(r * s * sin(alpha + theta_r));
        if (xc >= 0 && xc < cuSourceParams.width && yc >= 0 && yc < cuSourceParams.height) {
            int accumulatorIndex = is * nRotationSlices * (cuSourceParams.height / blockSize + 1) * (cuSourceParams.width / blockSize + 1) 
                                   + itheta * (cuSourceParams.height / blockSize + 1) * (cuSourceParams.width / blockSize + 1)
                                   + yc / blockSize * (cuSourceParams.width / blockSize + 1)
                                   + xc / blockSize;
            atomicAdd(accumulator + accumulatorIndex, 1);
        }
    }
}

__global__ void accumulate_kernel_3D_sharedMemory(int* accumulator, float* edgePixels, float* srcOrient, int* blockStarts, int* blockEnds) {
    int blockIndex = blockIdx.x;
    int start = blockStarts[blockIdx];
    int end = blockEnds[blockIdx];

    int threadIndex = threadIdx.x;
    int index = start + threadIndex;
    if (index > end) return;

    int itheta = blockIdx.y;
    int is = blockIdx.z;

    int pixelIndex = static_cast<int>(edgePixels[index]);
    float phi = srcOrient[pixelIndex]; // gradient direction in [0,360)
    float theta = itheta * deltaRotationAngle;
    float theta_r = theta / 180.f * PI;

    // minus mean rotate back by theta
    int iSlice = static_cast<int>(fmod(phi - theta + 360, 360) / deltaRotationAngle);
    
    float s = is * deltaScaleRatio + MINSCALE;

    __shared__ int accumulatorSlice[(cuSourceParams.height / blockSize + 1) * (cuSourceParams.width / blockSize + 1)];
    
    // access RTable and traverse all entries
    // std::vector<rEntry> entries = rTable[iSlice];
    for (auto entry: entries){
        float r = entry.r;
        float alpha = entry.alpha;
        int xc = i + round(r * s * cos(alpha + theta_r));
        int yc = j + round(r * s * sin(alpha + theta_r));
        if (xc >= 0 && xc < cuSourceParams.width && yc >= 0 && yc < cuSourceParams.height) {
            // int accumulatorIndex = is * nRotationSlices * (cuSourceParams.height / blockSize + 1) * (cuSourceParams.width / blockSize + 1) 
            //                        + itheta * (cuSourceParams.height / blockSize + 1) * (cuSourceParams.width / blockSize + 1)
            //                        + yc / blockSize * (cuSourceParams.width / blockSize + 1)
            //                        + xc / blockSize;
            // atomicAdd(accumulator + accumulatorIndex, 1);
            int accumulatorSliceIndex = yc / blockSize * (cuSourceParams.width / blockSize + 1) + xc / blockSize;
            atomicAdd(accumulatorSlice + accumulatorSliceIndex , 1);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        for (int j = 0; j < (cuSourceParams.height / blockSize + 1); j++) {
            for (int i = 0; i < (cuSourceParams.width / blockSize + 1); i++) {
                int accumulatorIndex = is * nRotationSlices * (cuSourceParams.height / blockSize + 1) * (cuSourceParams.width / blockSize + 1) 
                                       + itheta * (cuSourceParams.height / blockSize + 1) * (cuSourceParams.width / blockSize + 1)
                                       + j * (cuSourceParams.width / blockSize + 1)
                                       + i;
                int accumulatorSliceIndex = j * (cuSourceParams.width / blockSize + 1) + i;
                accumulator[accumulatorIndex] += accumulatorSlice[accumulatorSliceIndex];
            }
        }
    }
}

__global__ void findmaxima_kernel(int* accumulator, Point* blockMaxima) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int row = index / (cuSourceParams.width / blockSize + 1);
    int col = index % (cuSourceParams.width / blockSize + 1);

    int max = -1;
    blockMaxima[index].hits = 0;
    for (int itheta = 0; itheta < nRotationSlices; itheta++) {
        for (int is = 0; is < nScaleSlices; is++) {
            int accumulatorIndex = is * nRotationSlices * (cuSourceParams.height / blockSize + 1) * (cuSourceParams.width / blockSize + 1) 
                                   + itheta * (cuSourceParams.height / blockSize + 1) * (cuSourceParams.width / blockSize + 1)
                                   + row * (cuSourceParams.width / blockSize + 1)
                                   + col;
            if (accumulator[accumulatorIndex] > max) {
                max = accumulator[accumulatorIndex];
                blockMaxima[index].hits = max;
                blockMaxima[index].x = col * blockSize + blockSize / 2;
                blockMaxima[index].y = row * blockSize + blockSize / 2;
                blockMaxima[index].scale = is * deltaScaleRatio + MINSCALE;
                blockMaxima[index].rotation = itheta * deltaRotationAngle;
            }
        }
    }
}

// srcThreshold, srcOrient: pointers to GPU memory address
void CudaGeneralHoughTransform::accumulate(float* srcThreshold, float* srcOrient) {
    // Access using the following pattern:
    // sizeof(accumulator) = arbitrary * sizez * sizey * sizex
    // accumulator[l][k][j][i] = accumulator1D[l*(sizex*sizey*sizez) + k*(sizex*sizey) + j*(sizex) + i]
    int* accumulator;
    Point* blockMaxima;
    Point* hitPointsCuda;

    int sizeAccumulator = nScaleSlices * nRotationSlices * (src->height / blockSize + 1) * (src->width / blockSize + 1);
    cudaMalloc(&accumulator, sizeof(int) * sizeAccumulator);
    cudaMemset(accumulator, 0.f, sizeof(int) * sizeAccumulator);
    cudaMalloc(&blockMaxima, sizeof(Point) * (src->height / blockSize + 1) * (src->width / blockSize + 1));

    // Get all edge pixels
    // TODO: magThreshold put index instead of 255 at each element, -1 if not an edge pixel
    float* edgePixels; // expected data example: [1.0, 3.0, 7.0, ...] (float as it is copied from srcThreshold)
    cudaMalloc(&edgePixels, sizeof(float) * src->height * src->width);
    cudaMemset(edgePixels, 0.f, sizeof(int) * src->height * src->width);
    thrust::device_ptr<float> srcThresholdThrust = thrust::device_pointer_cast(srcThreshold); 
    thrust::device_ptr<float> edgePixelsThrust = thrust::device_pointer_cast(edgePixels); 
    int numEdgePixels = thrust::copy_if(srcThresholdThrust, srcThresholdThrust + src->width * src->height, edgePixelsThrust, is_edge()) - edgePixelsThrust;
    
    // --------------------------------------------------------------------
    // (1) naive approach: 1D partition -> divergent control flow on entries
    // Each block (e.g. 32 threads) take a part of the edge points
    // Each thread take 1 edge point
    // Write to global CUDA memory atomically
    int threadsPerBlock = 32;
    int blocks = (numEdgePixels + threadsPerBlock - 1) / threadsPerBlock;
    accumulate_kernel_naive<<<blocks, threadsPerBlock>>>(accumulator, edgePixels, numEdgePixels, srcOrient);
    cudaDeviceSynchronize();
    // --------------------------------------------------------------------

    // --------------------------------------------------------------------
    // (2) better approach: 
    //     a. put edge points into buckets by phi
    //     b. points with the same phi go together in a kernel (1D) each block has the same phi value
    
    // sort edgePixels by angle
    thrust::sort(edgePixelsThrust, edgePixelsThrust + numEdgePixels, compare_pixel_by_orient(srcOrient));

    // traverse sorted edge pixels and find the seperation points O(N)
    // Possible optimization: binary search O(logN) to find all the seperation points (72), can parallel the bineary search
    // Use 1 kernel to calculate the following:
    // GPU memory: int[nRotationSlices] (start index for each slice)
    // CPU memory: number of blocks (calculate using threadsPerBlock)
    int* startIndex;
    cudaMalloc(&startIndex, sizeof(int) * nRotationSlices);
    int* numBlocksDevice;
    cudaMalloc(&numBlocks, sizeof(int));
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

    accumulate_kernel_better<<<numBlocks, threadsPerBlock>>>(accumulator, edgePixels, srcOrient, blockStarts, blockEnds);
    cudaDeviceSynchronize();

    // (3) more parallelism: 
    //     a. put edge points into buckets by phi
    //     b. go by same phi, then same theta (2D) -> shared slice of 3D accumulator || then same scale (3D?) -> shared slice of 2D accumulator
    dim3 gridDim(numBlocks, nRotationSlices, nScaleSlices);
    accumulate_kernel_3D<<<gridDim, threadsPerBlock>>>(accumulator, edgePixels, srcOrient, blockStarts, blockEnds);
    cudaDeviceSynchronize();

    // shared memory of accumulator slice
    accumulate_kernel_3D_sharedMemory<<<gridDim, threadsPerBlock>>>(accumulator, edgePixels, srcOrient, blockStarts, blockEnds);
    cudaDeviceSynchronize();

    cudaFree(startIndex);
    cudaFree(blockStarts);
    cudaFree(blockEnds);
    // --------------------------------------------------------------------

    // Some high level notes: 
    // a. use atomic add when update accumulator
    // b. any idea to reduce global memory access?

    // Use a seperate kernel to fill in the blockMaxima array (avoid extra memory read)
    blocks = ((src->height / blockSize + 1) * (src->width / blockSize + 1) + threadsPerBlock - 1) / threadsPerBlock;
    findmaxima_kernel<<<blocks, threadsPerBlock>>>(accumulator, blockMaxima);
    cudaDeviceSynchronize();

    // use thrust::max_element to get the max hit from blockMaxima
    thrust::device_ptr<Point> blockMaximaThrust = thrust::device_pointer_cast(blockMaxima);
    Point maxPoint = *(thrust::max_element(blockMaximaThrust, blockMaximaThrust + (src->height / blockSize + 1) * (src->width / blockSize + 1), compare_point()));
    int maxHit = maxPoint.hits;

    // use thrust::copy_if to get the above threshold points & number
    cudaMalloc(hitPointsCuda, sizeof(Point) * numEdgePixels);
    thrust::device_ptr<Point> hitPointsThrust = thrust::device_pointer_cast(hitPointsCuda);
    int numResPoints = thrust::copy_if(blockMaximaThrust, blockMaximaThrust + (src->height / blockSize + 1) * (src->width / blockSize + 1), hitPointsThrust, filter_point(round(maxHit * thresRatio))) - blockMaximaThrust;

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
    // printf("max value in accumulator: %d\n", _max);
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
    // Each kernel is 1D
    // Each block (e.g. 512 threads) checks if the element is 255 (>254), and mark the position as 1
    // Use scan to get the index of the elements and the size
    // ...
    // Use thrust::copy_if to avoid all the hassles
    // thrust::copy_if(thresholdImage.begin(), thresholdImage.end(), target.begin(), predicator (>254))
    // where thresholdImage is on GPU, target is on GPU
    // then can get the number from (result (T: thrust::device_ptr) - target)
    // TODO: magThreshold put index instead of 255 at each element, -1 if not an edge pixel

    // (1) naive approach: 1D partition -> divergent control flow on entries
    // Each block (e.g. 32 threads) take a part of the edge points
    // Write to global CUDA memory atomically
    // Use a seperate kernel to check local maxima (avoid extra memory read)

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
