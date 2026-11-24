# Project Progress Log

## November 16, 2026

### Completed

- Set up the VLAD image search project in VS Code.
- Configured CMake and OpenCV 4.5.5.
- Implemented image loading and grayscale conversion.
- Implemented SIFT keypoint detection and descriptor extraction.
- Tested the implementation using a sample photograph.

### Results

- Keypoints detected: 616
- Descriptor matrix: 616 x 128
- Output: results/sift_keypoints.jpg

### Next Steps

1. Process multiple images.
2. Collect SIFT descriptors for clustering.
3. Implement k-means to generate a visual vocabulary.
4. Begin VLAD encoding.

## November 23, 2026

### Completed

- Reviewed the selected VLAD research paper to understand its methodology and limitations.
- Brainstormed potential improvements to the original approach with my teammate.
- Finalized geometric verification as an extension to the VLAD retrieval method.
- Designed a two-stage image retrieval pipeline using VLAD and RANSAC-based geometric verification.
- Identified the required OpenCV functions, implementation classes, and data structures.
- Planned the dataset selection, testing methodology, and evaluation metrics.
- Prepared the project design review presentation.

### Results

- Finalized the proposed approach and technical design.
- Established implementation responsibilities within the team.
- Defined experiments to evaluate retrieval accuracy, query time, and memory usage.
- Completed the project design review slides.

### Next Steps

1. Prepare the datasets for training and testing.
2. Extract SIFT descriptors from multiple images.
3. Implement k-means to generate a visual vocabulary.
4. Implement VLAD encoding and normalization.
5. Begin building the image index and retrieval pipeline.
