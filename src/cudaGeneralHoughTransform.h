#pragma once
#include <vector>

#include "generalHoughTransform.h"
#include "utils.h"

class CudaGeneralHoughTransform : public GeneralHoughTransform {
private:

public:
    CudaGeneralHoughTransform();

    virtual ~CudaGeneralHoughTransform();

    void setup() override;
    
    void processTemplate() override;

    void accumulateSource() override;

    void saveOutput() override;

    bool loadTemplate(std::string filename) override;

    bool loadSource(std::string filename) override;

private:
    Image* tpl;
    Image* src;
    int centerX, centerY;
    std::vector<Point> hitPoints;
    std::vector<std::vector<rEntry>> rTable;

    void convertToGray(const Image* image, GrayImage* result);
    
    void convolve(std::vector<std::vector<int>> filter, const GrayImage* source, GrayImage* result);

    void magnitude(const GrayImage* gradientX, const GrayImage* gradientY, GrayImage* result);

    void orientation(const GrayImage* gradientX, const GrayImage* gradientY, GrayImage* result);

    void edgenms(const GrayImage* magnitude, const GrayImage* orientation, GrayImage* result);

    void threshold(const GrayImage* magnitude, GrayImage* result, int threshold);

    void createRTable(const GrayImage* orientation, const GrayImage* magnitude);

};
