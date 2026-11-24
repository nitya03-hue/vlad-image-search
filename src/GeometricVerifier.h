// GeometricVerifier.h
// Stage 2 of our retrieval system: re-checks VLAD's top candidates with
// keypoint matching + RANSAC, and re-ranks them by geometric consistency.
//
// Author: Meghana Reddy Gavinolla

#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

// Everything we learn from checking one query/candidate pair.
struct VerificationResult {
    int numGoodMatches = 0;            // matches that passed the ratio test
    int numInliers = 0;                // matches that agree with the fitted model
    double inlierRatio = 0.0;          // numInliers / numGoodMatches
    bool validModel = false;           // a model was found with enough inliers
    double score = 0.0;                // what we sort by (higher = better)
    double millis = 0.0;               // time spent on this pair
    cv::Mat model;                     // 3x3 homography (or fundamental matrix)
    std::vector<cv::DMatch> goodMatches;
    std::vector<unsigned char> inlierMask;  // 1 = inlier, 0 = outlier (same order as goodMatches)
};

// One VLAD candidate handed to the verifier.
struct Candidate {
    int imageId = -1;
    double vladDistance = 0.0;         // from stage 1 (smaller = more similar)
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;               // N x 128, CV_32F
};

// A re-ranked result.
struct RankedResult {
    int imageId = -1;
    double vladDistance = 0.0;
    VerificationResult verification;
};

class GeometricVerifier {
public:
    enum class Model { Homography, Fundamental };

    struct Params {
        float ratio = 0.75f;           // Lowe's ratio test threshold
        double ransacThreshold = 5.0;  // max reprojection error in pixels
        int minInliers = 8;            // below this, the model is not trusted
        Model model = Model::Homography;
    };

    GeometricVerifier();
    explicit GeometricVerifier(const Params& params);

    // Step 1: match descriptors with knnMatch (k = 2) and keep matches
    // that pass Lowe's ratio test.
    std::vector<cv::DMatch> matchWithRatioTest(const cv::Mat& queryDesc,
                                               const cv::Mat& candDesc) const;

    // Steps 1-3: match, fit a model with RANSAC, and score one pair.
    VerificationResult verify(const std::vector<cv::KeyPoint>& queryKps,
                              const cv::Mat& queryDesc,
                              const std::vector<cv::KeyPoint>& candKps,
                              const cv::Mat& candDesc) const;

    // Step 4: verify every candidate and sort by score (high to low).
    // Ties keep their original VLAD order (stable sort).
    std::vector<RankedResult> rerank(const std::vector<cv::KeyPoint>& queryKps,
                                     const cv::Mat& queryDesc,
                                     const std::vector<Candidate>& candidates) const;

    // Draws match lines: inliers in green, outliers in red.
    static cv::Mat drawResult(const cv::Mat& queryImg,
                              const std::vector<cv::KeyPoint>& queryKps,
                              const cv::Mat& candImg,
                              const std::vector<cv::KeyPoint>& candKps,
                              const VerificationResult& result);

    const Params& params() const { return params_; }

private:
    Params params_;
};
