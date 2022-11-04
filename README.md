# Parallel Hough Circle Detector

This is 2022 Fall 15-618 Final Project by Yiliu Xu(yiliux), Yijie Chen(yijieche).

## SUMMARY
We are going to implement a parallel circle detector using Hough Transform on GPU. 

## BACKGROUND
Hough Transform is an image processing technique used for detection of parametric shapes. 

We choose to implement circle detection with arbitary radius, as we believe an efficient parallel implementation for circle detection will be valuable in a wide range of areas, such as medical image processing, and robot vision.

## THE CHALLENGE
Circles construct a 3D parameterized Hough space (a,b,r). Compared with line detection which forms a 2D space, the circle detection problem adds one more degree of computation complexity. In addition, it requires a large memory capacity to store the accumulator in 3D space. 

## RESOURCES


## GOALS AND DELIVERABLES


## PLATFORM CHOICE


## SCHEDULE
