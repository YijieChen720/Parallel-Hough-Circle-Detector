#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>

#include "generalHoughTransform.h"
#include "seqGeneralHoughTransform.h"
#include "cudaGeneralHoughTransform.h"
#include "utils.h"
#include "cycleTimer.h"

void usage(const char* progname) {
    printf("Usage: %s -t template -s source -r <seq/cuda> [options]\n", progname);
    printf("Program Options:\n");
    printf("  -a  --strategy <naive/sort/binning>  Select strategy: naive, sort or binning\n");
    printf("  -d  --dimension <1/3>  Select kernel dimension: 1D or 3D\n");
    printf("  -?  --help                 This message\n");
}

int main(int argc, char** argv) {
    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"help",      0, 0,  'h'},
        {"template",  1, 0,  't'},
        {"source",    1, 0,  's'},
        {"renderer",  1, 0,  'r'},
        {"strategy",  1, 0,  'a'},
        {"dimension", 1, 0,  'd'},
        {0 ,0, 0, 0}
    };
    
    GeneralHoughTransform* ght;
    std::string templateFilename, sourceFilename;

    // By default using the optimal implementation
    bool naive = false;
    bool sort = true;
    bool is1D = false;

    while ((opt = getopt_long(argc, argv, "t:s:r:a:d:h", long_options, NULL)) != EOF) {
        switch (opt) {
        case 't':
            // Store template filename
            templateFilename = std::string(optarg);
            printf("Template filename: %s\n", std::string(optarg).c_str());
            break;
        case 's':
            // Store source filename
            sourceFilename = std::string(optarg);
            printf("Source filename: %s\n", std::string(optarg).c_str());
            break;
        case 'r':
            if (std::string(optarg).compare("cuda") == 0) {
                printf("Using cuda implementation\n");
                ght = new CudaGeneralHoughTransform();
            } else {
                printf("Using sequential implementation\n");
                ght = new SeqGeneralHoughTransform();
            }
            break;
        case 'a':
            if (std::string(optarg).compare("naive") == 0) {
                printf("With naive parallelization strategy\n");
                naive = true;
                sort = false;
            } else if (std::string(optarg).compare("sort") == 0) {
                printf("With sort parallelization strategy\n");
                naive = false;
                sort = true;
            } else if (std::string(optarg).compare("binning") == 0) {
                printf("With binning parallelization strategy\n");
                naive = false;
                sort = false;
            }
            break;
        case 'd':
            if (std::string(optarg).compare("1") == 0) {
                printf("Accumulation kernel is 1D\n");
                is1D = true;
            } else if (std::string(optarg).compare("3") == 0) {
                printf("Accumulation kernel is 3D\n");
                is1D = false;
            }
            break;
        case 'h':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////

    // Image* templateImage = new Image;
    // readPPMImage(templateFilename, templateImage);
    // writePPMImage(templateImage, "test.ppm");


    std::cout << "***Loading images.***\n";
    if (!ght->loadTemplate(templateFilename) || !ght->loadSource(sourceFilename)) {
        std::cerr << "***Failed to load images.***\n";
        return 1;
    }
    std::cout << "***Finished loading images.***\n";

    ght->setup();

    double startTemplateTime = CycleTimer::currentSeconds();
    ght->processTemplate();
    double endTemplateTime = CycleTimer::currentSeconds();
    
    double startSourceTime = CycleTimer::currentSeconds();
    ght->accumulateSource(naive, sort, is1D);
    double endSourceTime = CycleTimer::currentSeconds();
    
    ght->saveOutput();

    delete ght;

    double totalTemplateTime = endTemplateTime - startTemplateTime;
    double totalSourceTime = endSourceTime - startSourceTime;

    printf("Process Template:  %.4f ms\n", 1000.f * totalTemplateTime);
    printf("Process Source:    %.4f ms\n", 1000.f * totalSourceTime);
}