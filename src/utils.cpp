#include "utils.h"
#include <iostream>

// To extract value from header
int getValue(char *line, int &pos);

unsigned char clamp(float value, unsigned char minV, unsigned char maxV) {
    if (value < minV) return minV;
    if (value > maxV) return maxV;
    return static_cast<unsigned char>(value);
}

// Only supports P6 with 255 color range
// Code borrowed from: https://www.delftstack.com/howto/cpp/read-ppm-file-cpp/
bool readPPMImage(std::string filename, Image* result) {
    FILE *read = fopen(filename.c_str(),"rb");
    if (!read) {
        std::cerr << "Unable to open the file " << filename << "\n";
        return false;
    }
    std::cout << "Loading Image: " << filename << std::endl;

    char magicNumber[256];
    fgets(magicNumber, 256, read);
    std::cout << "Magic Number: " << magicNumber;
    if (magicNumber[0] != 'P' || magicNumber[1] != '6'){
        std::cerr << "File format not supported\n";
        return false;
    }

    char wh[256];
    fgets(wh, 256, read);
    int pos = 0;
    int width = getValue(wh, pos);
    pos++;
    int height = getValue(wh, pos);
    std::cout << "Width:" << width << "\tHeight:" << height << '\n';

    char maxColorValueL[256];
    fgets(maxColorValueL, 256, read);
    pos = 0;
    int maxColorValue = getValue(maxColorValueL, pos);
    std::cout << "Max Color Value:" << maxColorValue << '\n';
    if (maxColorValue != 255) {
        std::cerr << "File format not supported\n";
        return false;
    }

    result->setImage(width, height);
    fread(result->data, 3 * width * height, 1, read);
    fclose(read);

    return true;
}

void writePPMImage(const Image* image, std::string filename) {
    FILE *fp = fopen(filename.c_str(), "wb");

    if (!fp) {
        fprintf(stderr, "Error: could not open %s for write\n", filename.c_str());
        exit(1);
    }

    // write ppm header
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", image->width, image->height);
    fprintf(fp, "255\n");

    fwrite(image->data, 3 * image->width * image->height, 1, fp);

    fclose(fp);
    printf("Wrote image file %s\n", filename.c_str());
}

void writeGrayPPMImage(const GrayImage* image, std::string filename) {
    FILE *fp = fopen(filename.c_str(), "wb");

    if (!fp) {
        fprintf(stderr, "Error: could not open %s for write\n", filename.c_str());
        exit(1);
    }

    // write ppm header
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", image->width, image->height);
    fprintf(fp, "255\n");

    unsigned char* colored = new unsigned char[3 * image->width * image->height];
    for (int i = 0; i < image->width * image->height; i++) {
        colored[i * 3] = clamp(image->data[i], 0, 255);
        colored[i * 3 + 1] = clamp(image->data[i], 0, 255);
        colored[i * 3 + 2] = clamp(image->data[i], 0, 255);
    }
    fwrite(colored, 3 * image->width * image->height, 1, fp);
    delete[] colored;

    fclose(fp);
    printf("Wrote gray scale image file %s\n", filename.c_str());
}

int getValue(char *line, int &pos) {
    int res = 0;
    for (; line[pos] != '\n' && line[pos] != ' ' && line[pos] != '\0'; pos++)
        res = res * 10 + (line[pos] - '0');
    return res;
}

bool keepPixel(const GrayImage* magnitude, int i, int j, int gradient) {
    int neighbourOnei = i;
    int neighbourOnej = j;
    int neighbourTwoi = i;
    int neighbourTwoj = j;
    
    switch (gradient) {
    case 0:
        neighbourOnei -= 1;
        neighbourTwoi += 1;
        break;
    case 45:
        neighbourOnej -= 1;
        neighbourOnei += 1;
        neighbourTwoj += 1;
        neighbourTwoi -= 1;
        break;
    case 90:
        neighbourOnej -= 1;
        neighbourTwoj += 1;
        break;
    default: // 135
        neighbourOnej -= 1;
        neighbourOnei -= 1;
        neighbourTwoj += 1;
        neighbourTwoi += 1;
    }
    
    float neighbourOne, neighbourTwo;
    if (neighbourOnei < 0 || neighbourOnei >= magnitude->width || neighbourOnej < 0 || neighbourOnej >= magnitude->height) neighbourOne = 0.f;
    else neighbourOne = magnitude->data[neighbourOnej * magnitude->width + neighbourOnei];
    if (neighbourTwoi < 0 || neighbourTwoi >= magnitude->width || neighbourTwoj < 0 || neighbourTwoj >= magnitude->height) neighbourTwo = 0.f;
    else neighbourTwo = magnitude->data[neighbourTwoj * magnitude->width + neighbourTwoi];
    float cur = magnitude->data[j * magnitude->width + i];
    
    return (neighbourOne <= cur) && (neighbourTwo <= cur);
}