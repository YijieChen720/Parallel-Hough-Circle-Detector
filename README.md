# Parallel Hough Circle Detector

This is 2022 Fall 15-618 Final Project by Yiliu Xu(yiliux), Yijie Chen(yijieche).

## SUMMARY
We are going to implement a parallel circle detector using Hough Transform on GPU. 

## BACKGROUND
Hough Transform is an image processing technique used for detection of parametric shapes. 

We choose to implement circle detection with arbitary radius, as we believe an efficient parallel implementation for circle detection will be valuable in a wide range of areas, such as medical image processing, and robot vision.

The pipeline of implementing a Hough Circle Detector is as follows:
- Apply a derivative filter (e.g. Sobel filter) to convert the raw rgb image into a binary image, '1' representing an edge pixel
- Apply Hough Transform and accumuate the vote of each edge pixel to a 3D accumulator matrix
- (Optional: use the gradient image to narrow down the voting space)
- Traverse the accumulator matrix and find the local maxima

## THE CHALLENGE
Circles construct a 3D parameterized Hough space (a,b,r). Compared with line detection which forms a 2D space, the circle detection problem adds one more degree of computation complexity. In addition, it requires a large memory capacity to store the 3D accumulator. It's challenging to design and deploy CUDA program to achieve high performance, memory efficiency and reduced communication. 

## RESOURCES
The starter C++ code will be structured from the current Python code of 16-720 Computer Vision Assignment 1. Note that the Assignment code implements a Hough line detector, so necessary modification is required to extend it to a circle detector. 

Some other research papers we will look into:
- Chen, Su, and Hai Jiang. "[Accelerating the hough transform with CUDA on graphics processing units.](http://worldcomp-proceedings.com/proc/p2011/PDP4179.pdf)"
- Askari, Meisam, et al. "Parallel gpu implementation of hough transform for circles." 
- Tasel, Faris Serdar, and Alptekin Temizel. "[Parallelization of hough transform for circles using cuda.](https://on-demand.gputechconf.com/gtc/2012/posters/P0438_ht_poster_gtc2012.pdf)" 

## GOALS AND DELIVERABLES
- 75% Implement a working version of CUDA circle detector with known radius (2D accumulator matrix)

- 100% Implement a working version of CUDA circle detector with arbitary radius. Note each part of the pipeline will need to be highly parallelized.

- 125% 

## PLATFORM CHOICE
We choose to implement this project using C++ on CUDA, as it's an image processing problem that fits well with GPU's architecture feature of high parallelization.

## SCHEDULE
