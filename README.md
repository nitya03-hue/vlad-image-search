# Image Search Using VLAD

**Course:** CSS 587 - Computer Vision  
**Student:** Nityasree Bellamkonda  
**Semester:** Autumn 2026

## Project Overview

This project implements an image retrieval system using
Vector of Locally Aggregated Descriptors (VLAD).

The goal is to retrieve the five most similar images
from a dataset given a query image.

## Research Paper

Jégou et al., "Aggregating Local Descriptors into a
Compact Image Representation," CVPR 2010.

## Implementation Plan

1. Image loading and preprocessing
2. SIFT feature extraction
3. K-means visual vocabulary generation
4. VLAD descriptor encoding
5. Image similarity calculation
6. Top-5 image retrieval
7. Performance evaluation and comparison

## Current Progress

### November 16, 2026

- Created the C++ project structure.
- Configured CMake with OpenCV 4.5.5.
- Implemented SIFT feature extraction.
- Successfully detected 616 keypoints in a test image.
- Generated 616 descriptors, each with 128 dimensions.
- Saved a visualization of the detected keypoints.

## Initial Result

![SIFT Keypoints](results/sift_keypoints.jpg)

## Build Instructions

Configure the project:

    cmake -S . -B build -DOpenCV_DIR=/Users/nityabellamkonda/opencv-4.5.5-install/lib/cmake/opencv4

Build the program:

    cmake --build build

Run:

    ./build/vlad_search

## Next Steps

- Extract SIFT descriptors from multiple images.
- Implement k-means clustering.
- Generate VLAD representations.
- Begin testing image retrieval.

