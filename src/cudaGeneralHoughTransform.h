#pragma once

#include "generalHoughTransform.h"

class CudaGeneralHoughTransform : public GeneralHoughTransform {
private:

public:
    CudaGeneralHoughTransform();

    virtual ~CudaGeneralHoughTransform();

private:
    // void convolve(int **filter, size_t k, const GrayImage* source, GrayImage& result) override;
};
