#pragma once

#include "generalHoughTransform.h"

class CudaGeneralHoughTransform : public GeneralHoughTransform {
private:
    Image* tpl, src;

public:
    CudaGeneralHoughTransform();

    virtual ~CudaGeneralHoughTransform();

private:
    void convolve(int **filter, size_t k, Image& result);
};
