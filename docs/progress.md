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

## November 23, 2026 (Meghana Reddy Gavinolla)

### Completed

- Implemented the `GeometricVerifier` class for stage 2 (re-ranking):
  - Descriptor matching with `knnMatch` (k = 2) and Lowe's ratio test (0.75).
  - Model fitting with RANSAC: `cv::findHomography` by default, `cv::findFundamentalMat` as an option for 3D scenes.
  - Score = number of inliers when the model is valid (at least 8 inliers).
  - Re-ranking of candidates with a stable sort, so ties keep their VLAD order.
  - Visualization of matches (inliers green, outliers red).
- Added the `verify_pair` test program with three modes: `selftest`, `pair`, `rerank`.
- Added CMake targets `geometric_verifier` (library) and `verify_pair` (program).

### Results

| Test | Good matches | Inliers | Decision |
|---|---|---|---|
| Sample image vs. warped copy (15° rotation, 0.8 scale, perspective) | 243 | 227 | Accepted |
| Sample image vs. unrelated texture | 1 | 0 | Rejected |

- Re-ranking test: the unrelated image was placed first (as if VLAD got it wrong); after re-ranking, the correct image moved to rank 1.
- Output images: `results/selftest_same.jpg`, `results/selftest_unrelated.jpg`.

### Next Steps

1. Run `verify_pair` on real Holidays pairs (same scene and different scenes).
2. Tune the ratio threshold and RANSAC threshold.
3. Compare homography vs. fundamental matrix.
4. Connect the verifier to the VLAD top-20 list after VLAD encoding is ready.
