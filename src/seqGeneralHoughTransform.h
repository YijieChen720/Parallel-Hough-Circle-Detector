#pragma once

#include "generalHoughTransform.h"

class SeqGeneralHoughTransform : public GeneralHoughTransform {
private:
    Image* tpl, src;

public:
    SeqGeneralHoughTransform();

    virtual ~SeqGeneralHoughTransform();

private:
    void convolve(int **filter, size_t k, Image& result);
};
