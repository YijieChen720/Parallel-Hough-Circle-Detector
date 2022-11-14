#pragma once
#include <vector>

#include "generalHoughTransform.h"
#include "utils.h"

class SeqGeneralHoughTransform : public GeneralHoughTransform {
private:

public:
    SeqGeneralHoughTransform();

    ~SeqGeneralHoughTransform();

    void processTemplate() override;

    void accumulateSource() override;

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

    void threshold(const GrayImage* magnitude, GrayImage* result, int threshold);

    void createRTable(const GrayImage* orientation, const GrayImage* magnitude);

    bool localMaxima(std::vector<std::vector<Point>> blockMaxima, int i, int j, int maximaThres);
};
