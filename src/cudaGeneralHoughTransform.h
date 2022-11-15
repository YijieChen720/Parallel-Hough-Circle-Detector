#pragma once

#include "generalHoughTransform.h"

class CudaGeneralHoughTransform : public GeneralHoughTransform {
private:

public:
    CudaGeneralHoughTransform();

    virtual ~CudaGeneralHoughTransform();

    void processTemplate() override;

    void accumulateSource() override;

    void saveOutput() override;

    bool loadTemplate(std::string filename) override;

    bool loadSource(std::string filename) override;

private:
    // void convolve(int **filter, size_t k, const GrayImage* source, GrayImage& result) override;
};
