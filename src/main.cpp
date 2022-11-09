#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "generalHoughTransform.h"
#include "seqGeneralHoughTransform.h"
// #include "cudaGeneralHoughTransform.h"
#include "utils.h"

void usage(const char* progname) {
    printf("Usage: %s -t template -s source [options]\n", progname);
    printf("Program Options:\n");
    printf("  -r  --renderer <seq/cuda>  Select renderer: seq or cuda\n");
    printf("  -?  --help                 This message\n");
}

int main(int argc, char** argv) {
    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"help",     0, 0,  'h'},
        {"template", 1, 0,  't'},
        {"source",   1, 0,  's'},
        {"renderer", 1, 0,  'r'},
        {0 ,0, 0, 0}
    };
    
    GeneralHoughTransform* hgt;
    std::string templateFilename, sourceFilename;

    while ((opt = getopt_long(argc, argv, "t:s:r:h", long_options, NULL)) != EOF) {
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
            if (std::string(optarg).compare("seq") == 0) {
                printf("Using sequential implementation\n");
                // hgt = new SeqGeneralHoughTransform();
            } else if (std::string(optarg).compare("cuda") == 0) {
                printf("Using cuda implementation\n");
                // hgt = new CudaGeneralHoughTransform();
            }
            break;
        case 'h':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////

    Image templateImage;
    readPPMImage(templateFilename, templateImage);
    // writePPMImage(&templateImage, "test.ppm");
}