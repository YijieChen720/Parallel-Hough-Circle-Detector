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
    std::vector<std::vector<rEntry>> rTable;

    void convolve(int **filter, size_t k, const GrayImage* source, GrayImage& result) override;
};
