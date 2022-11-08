#pragma once

class Image {
private:
    int width;
    int height;
    float* data;
    
public:
    Image () {}

    Image (int w, int h) {
        width = w;
        height = h;
        data = new float[4 * width * height];
    }
};
