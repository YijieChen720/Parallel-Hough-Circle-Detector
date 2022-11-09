#pragma once
#include <vector>

class Image {
public:
    Image () {}

    void setImage (int w, int h) {
        width = w;
        height = h;

        data = new unsigned char[3 * width * height];
    }

    int width;
    int height;
    unsigned char* data; // 0-255
};
