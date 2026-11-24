// GeometricVerifier.cpp
// Author: Meghana Reddy Gavinolla

#include "GeometricVerifier.h"

#include <algorithm>
#include <chrono>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

GeometricVerifier::GeometricVerifier() : params_() {}

GeometricVerifier::GeometricVerifier(const Params& params) : params_(params) {}

std::vector<cv::DMatch> GeometricVerifier::matchWithRatioTest(const cv::Mat& queryDesc,
                                                              const cv::Mat& candDesc) const {
    std::vector<cv::DMatch> good;
    if (queryDesc.empty() || candDesc.rows < 2) return good;

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(queryDesc, candDesc, knn, 2);

    for (const auto& pair : knn) {
        // Keep a match only if the best is clearly better than the second best.
        if (pair.size() == 2 && pair[0].distance < params_.ratio * pair[1].distance) {
            good.push_back(pair[0]);
        }
    }
    return good;
}

VerificationResult GeometricVerifier::verify(const std::vector<cv::KeyPoint>& queryKps,
                                             const cv::Mat& queryDesc,
                                             const std::vector<cv::KeyPoint>& candKps,
                                             const cv::Mat& candDesc) const {
    auto start = std::chrono::steady_clock::now();
    VerificationResult r;

    // Step 1: match.
    r.goodMatches = matchWithRatioTest(queryDesc, candDesc);
    r.numGoodMatches = static_cast<int>(r.goodMatches.size());

    // A homography needs at least 4 point pairs, a fundamental matrix at least 8.
    const size_t minPoints = (params_.model == Model::Homography) ? 4 : 8;

    if (r.goodMatches.size() >= minPoints) {
        std::vector<cv::Point2f> qPts, cPts;
        qPts.reserve(r.goodMatches.size());
        cPts.reserve(r.goodMatches.size());
        for (const auto& m : r.goodMatches) {
            qPts.push_back(queryKps[m.queryIdx].pt);
            cPts.push_back(candKps[m.trainIdx].pt);
        }

        // Step 2: fit a model with RANSAC. The mask marks the inliers.
        cv::Mat mask;
        if (params_.model == Model::Homography) {
            r.model = cv::findHomography(qPts, cPts, cv::RANSAC, params_.ransacThreshold, mask);
        } else {
            r.model = cv::findFundamentalMat(qPts, cPts, cv::FM_RANSAC, params_.ransacThreshold, 0.99, mask);
        }

        if (!r.model.empty() && !mask.empty()) {
            r.inlierMask.assign(mask.begin<unsigned char>(), mask.end<unsigned char>());
            r.numInliers = cv::countNonZero(mask);
        }
    }

    // Step 3: score.
    r.inlierRatio = r.numGoodMatches > 0 ? static_cast<double>(r.numInliers) / r.numGoodMatches : 0.0;
    r.validModel = !r.model.empty() && r.numInliers >= params_.minInliers;
    // First version of the score: the inlier count, but only if the model is trusted.
    // (We will test whether adding the inlier ratio helps, using precision at 5.)
    r.score = r.validModel ? static_cast<double>(r.numInliers) : 0.0;

    r.millis = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return r;
}

std::vector<RankedResult> GeometricVerifier::rerank(const std::vector<cv::KeyPoint>& queryKps,
                                                    const cv::Mat& queryDesc,
                                                    const std::vector<Candidate>& candidates) const {
    std::vector<RankedResult> results;
    results.reserve(candidates.size());

    // Candidates arrive in VLAD order (best first).
    for (const auto& c : candidates) {
        RankedResult rr;
        rr.imageId = c.imageId;
        rr.vladDistance = c.vladDistance;
        rr.verification = verify(queryKps, queryDesc, c.keypoints, c.descriptors);
        results.push_back(rr);
    }

    // Step 4: re-rank by score. stable_sort keeps VLAD order for ties.
    std::stable_sort(results.begin(), results.end(), [](const RankedResult& a, const RankedResult& b) {
        return a.verification.score > b.verification.score;
    });
    return results;
}

cv::Mat GeometricVerifier::drawResult(const cv::Mat& queryImg,
                                      const std::vector<cv::KeyPoint>& queryKps,
                                      const cv::Mat& candImg,
                                      const std::vector<cv::KeyPoint>& candKps,
                                      const VerificationResult& result) {
    std::vector<cv::DMatch> inliers, outliers;
    for (size_t i = 0; i < result.goodMatches.size(); ++i) {
        bool isInlier = i < result.inlierMask.size() && result.inlierMask[i];
        (isInlier ? inliers : outliers).push_back(result.goodMatches[i]);
    }

    cv::Mat canvas;
    // Outliers first (red), then inliers on top (green).
    cv::drawMatches(queryImg, queryKps, candImg, candKps, outliers, canvas,
                    cv::Scalar(0, 0, 255), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::drawMatches(queryImg, queryKps, candImg, candKps, inliers, canvas,
                    cv::Scalar(0, 255, 0), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::DRAW_OVER_OUTIMG | cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    std::string label = "matches: " + std::to_string(result.numGoodMatches) +
                        "  inliers: " + std::to_string(result.numInliers) +
                        (result.validModel ? "  (valid)" : "  (rejected)");
    cv::putText(canvas, label, cv::Point(15, 35), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(255, 255, 255), 4, cv::LINE_AA);
    cv::putText(canvas, label, cv::Point(15, 35), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    return canvas;
}
