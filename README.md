# Parallel Object Detector with Generalized Hough Transform

This is 2022 Fall 15-618 Final Project by Yiliu Xu(yiliux), Yijie Chen(yijieche).

## SUMMARY
We are going to implement a parallel object detector using generalized Hough Transform on GPU. 

## BACKGROUND
Hough Transform is an image processing technique used for detecting parametric shapes, such as lines and circles. [A generalized version of Hough Transform](http://www.eng.tau.ac.il/~cvapps/Supplement/%5B%201981%20%5D%20Generalizing%20the%20Hough%20Transform%20to%20Detect%20Arbitrary%20Shapes.pdf) proposed by Ballard is able to detect any arbitrary shapes, even ones that cannot be described by equations, by encoding the input template shape and mapping the raw image into a Hough Transform space. We believe an efficient parallel implementation for generalized Hough detector will be valuable in a wide range of areas, such as medical image processing, and robot vision.

The pipeline of implementing a Generalized Hough Detector is as follows:
- Apply a derivative filter (e.g. Sobel filter) on both the template image and raw image to detect all edge pixels
- Encode all edge points of the template shape into an R-table
- Apply Generalized Hough Transform on the raw image, accumulate the vote of each edge pixel into a 4D accumulator matrix $(x_c, y_c, s, \theta)$
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

- 100% Implement a working version of CUDA circle detector with arbitrary radius. Note each part of the pipeline will need to be highly parallelized.

- 125% Implement the inverse-checkcing mapping strategy proposed in [Accelerating the hough transform with CUDA on graphics processing units](http://worldcomp-proceedings.com/proc/p2011/PDP4179.pdf) which is more space efficient.

## PLATFORM CHOICE
We choose to implement this project using C++ on CUDA, as it's an image processing problem that fits well with GPU's feature of massively parallel execution.

## SCHEDULE
11/06 - 11/12 Conduct literature review to better understand the problem. Environment setup on GHC machine. Implement the sequential version.  
11/13 - 11/19 Design parallel strategies for each component in the pipeline.  
11/20 - 11/26 Implement a parallel version in CUDA.  
11/27 - 12/03 Benchmark and optimize the parallel version.  
12/04 - 12/08 Write report and prepare for poster session.  
