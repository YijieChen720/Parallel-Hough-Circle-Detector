#pragma once
#include <iostream>

struct Image {
    Image() : width(0), height(0), data(nullptr) {}
    
    void setImage (int w, int h) {
        width = w;
        height = h;

        data = new unsigned char[3 * width * height];
    }

    int width;
    int height;
    unsigned char* data; // 0-255
};

struct GrayImage {
    GrayImage() : width(0), height(0), data(nullptr) {}

    void setGrayImage (int w, int h) {
        width = w;
        height = h;

        data = new unsigned char[width * height];
    }

    int width;
    int height;
    unsigned char* data; // 0-255
};