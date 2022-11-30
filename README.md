# Parallel Object Detector with Generalized Hough Transform

This is 2022 Fall 15-618 Final Project by Yiliu Xu(yiliux), Yijie Chen(yijieche).

## SUMMARY
We are going to implement a parallel object detector using generalized Hough Transform on GPU. 

## REPORTS
[Project Proposal](15618_Proposal.pdf)  
[Project Milestone Report](15618_Milestone.pdf)

## BACKGROUND
Hough Transform is an image processing technique used for detecting parametric shapes, such as lines and circles. [A generalized version of Hough Transform](http://www.eng.tau.ac.il/~cvapps/Supplement/%5B%201981%20%5D%20Generalizing%20the%20Hough%20Transform%20to%20Detect%20Arbitrary%20Shapes.pdf) proposed by Ballard is able to detect any arbitrary shapes, even ones that cannot be described by equations, by encoding the input template shape and mapping the raw image into a Hough Transform space. We believe an efficient parallel implementation for generalized Hough detector will be valuable in a wide range of areas, such as medical image processing, and robot vision.

The pipeline of implementing a Generalized Hough Detector is as follows:
- Apply a derivative filter (e.g. Sobel filter) on both the template image and raw image to detect all edge pixels
- Encode all edge points of the template shape into an R-table
- Apply Generalized Hough Transform on the raw image, accumulate the vote of each edge pixel into a 4D accumulator matrix $(x_c, y_c, s, \theta)$
- Traverse the accumulator matrix and find the local maxima

## THE CHALLENGE
We aim to construct a 4D parameterized Hough space (x, y, scale, rotation) to detect arbitrary input shapes. Above all, generalized hough algorithm requires more computation and processing for non-analytic curves than simple lines or circles. Besides, compared with the detection with invariant scale and rotation in 2D space, this detection problem adds two more degree of computation complexity. In addition, it requires a large memory capacity to store the 4D accumulator. It's challenging to design and deploy CUDA program to achieve high performance, memory efficiency and reduced communication. 

## RESOURCES
The starter C++ code will be structured from this [github repo](https://github.com/jguillon/generalized-hough-tranform). Note that the reference code uses OpenCV to compute image gradient, but we will write the convolution filter from stratch in order to parallelize the code using CUDA.

Some other research papers we will look into:
- Ballard, Dana H. "[Generalizing the Hough transform to detect arbitrary shapes.](https://www.sciencedirect.com/science/article/abs/pii/0031320381900091)"
- Chen, Su, and Hai Jiang. "[Accelerating the hough transform with CUDA on graphics processing units.](http://worldcomp-proceedings.com/proc/p2011/PDP4179.pdf)"

## UPDATED GOALS AND DELIVERABLES
- 75% Implement a working version of CUDA Generalized Hough detector and try our best to optimize it.

- 100% Implement multiple working versions of CUDA Generalized Hough detector with arbitrary scale and rotation. Conduct a detailed analysis on each parallel strategy. Note each part of the pipeline will need to be highly parallelized.

- 125% Optimize the Generalized Hough detector by using [Halide](https://halide-lang.org/) and improve detector's performance.

## PLATFORM CHOICE
We choose to implement this project using C++ on CUDA, as it's an image processing problem that fits well with GPU's feature of massively parallel execution.

## UPDATED SCHEDULE
11/06 - 11/12 Conduct literature review to better understand the problem. Environment setup on GHC machine. Implement the sequential version.  
11/13 - 11/19 Design parallel strategies for each component in the pipeline.  
11/20 - 11/26 Implement a parallel version in CUDA: Image Processing (Yiliu), Accumulator (Yijie)
11/27 - 11/30 Merge code and debug. Time each part of code. Generate a computationally expensive testcase. (Yiliu, Yijie)
12/1 - 12/03 If we are able to achieve expected speedup, start analyzing and comparing different parallelization approaches; otherwise keep debugging and optimizing our solution. (Yiliu, Yijie)
12/04 - 12/08 Write report and prepare for poster session. (Yiliu, Yijie)
