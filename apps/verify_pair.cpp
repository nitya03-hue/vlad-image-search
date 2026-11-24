// verify_pair.cpp
// Test program for the GeometricVerifier (Meghana's part).
//
// Usage:
//   verify_pair selftest <image>
//       Checks the verifier on a known case: the image vs. a warped copy of itself
//       (should be accepted) and vs. a random texture (should be rejected).
//
//   verify_pair pair <query> <candidate> [--model H|F] [--ratio 0.75] [--thresh 5] [--out results/matches.jpg]
//       Matches two images, runs RANSAC, prints the numbers and saves a picture
//       with inliers in green and outliers in red.
//
//   verify_pair rerank <query> <cand1> <cand2> ...
//       Treats the candidates as a VLAD shortlist (in the given order) and prints
//       the order before and after geometric re-ranking.
//
// Author: Meghana Reddy Gavinolla

#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "GeometricVerifier.h"

namespace {

// Holidays images are large; we shrink them so the longer side is at most 1024 px.
const int kMaxSize = 1024;

cv::Mat loadResized(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) return img;
    int longer = std::max(img.cols, img.rows);
    if (longer > kMaxSize) {
        double s = static_cast<double>(kMaxSize) / longer;
        cv::resize(img, img, cv::Size(), s, s, cv::INTER_AREA);
    }
    return img;
}

void extract(const cv::Mat& img, std::vector<cv::KeyPoint>& kps, cv::Mat& desc) {
    static cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    sift->detectAndCompute(gray, cv::noArray(), kps, desc);
    if (!desc.empty()) desc.convertTo(desc, CV_32F);
}

void printResult(const std::string& name, const VerificationResult& r) {
    std::cout << std::left << std::setw(28) << name
              << " good matches: " << std::setw(5) << r.numGoodMatches
              << " inliers: " << std::setw(5) << r.numInliers
              << " ratio: " << std::fixed << std::setprecision(2) << std::setw(6) << r.inlierRatio
              << (r.validModel ? " VALID   " : " rejected")
              << " time: " << std::setprecision(1) << r.millis << " ms" << std::endl;
}

std::map<std::string, std::string> parseOptions(int argc, char** argv, int start) {
    std::map<std::string, std::string> opts;
    for (int i = start; i + 1 < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) == 0) opts[key.substr(2)] = argv[++i];
    }
    return opts;
}

GeometricVerifier makeVerifier(const std::map<std::string, std::string>& opts) {
    GeometricVerifier::Params p;
    if (opts.count("ratio")) p.ratio = std::stof(opts.at("ratio"));
    if (opts.count("thresh")) p.ransacThreshold = std::stod(opts.at("thresh"));
    if (opts.count("model") && (opts.at("model") == "F" || opts.at("model") == "f"))
        p.model = GeometricVerifier::Model::Fundamental;
    return GeometricVerifier(p);
}

// ---------------- selftest ----------------
int runSelfTest(const std::string& path) {
    cv::Mat img = loadResized(path);
    if (img.empty()) { std::cerr << "Could not read " << path << std::endl; return 1; }

    // 1) A warped copy of the same image: rotate 15 degrees, scale 0.8, add perspective.
    cv::Point2f center(img.cols / 2.0f, img.rows / 2.0f);
    cv::Mat A = cv::getRotationMatrix2D(center, 15.0, 0.8);
    cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
    A.copyTo(H(cv::Rect(0, 0, 3, 2)));
    H.at<double>(2, 0) = 0.0002;  // a little perspective
    cv::Mat warped;
    cv::warpPerspective(img, warped, H, img.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // 2) An unrelated image: smoothed random texture.
    cv::Mat noise(img.size(), CV_8UC3);
    cv::randu(noise, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::GaussianBlur(noise, noise, cv::Size(7, 7), 2.0);

    std::vector<cv::KeyPoint> qk, wk, nk;
    cv::Mat qd, wd, nd;
    extract(img, qk, qd);
    extract(warped, wk, wd);
    extract(noise, nk, nd);
    std::cout << "Keypoints  query: " << qk.size() << "  warped: " << wk.size()
              << "  unrelated: " << nk.size() << std::endl;

    GeometricVerifier verifier;
    VerificationResult same = verifier.verify(qk, qd, wk, wd);
    VerificationResult diff = verifier.verify(qk, qd, nk, nd);
    printResult("same scene (warped copy)", same);
    printResult("unrelated image", diff);

    // Re-ranking check: put the unrelated image FIRST, as if VLAD got it wrong.
    std::vector<Candidate> cands(2);
    cands[0].imageId = 1; cands[0].vladDistance = 0.10; cands[0].keypoints = nk; cands[0].descriptors = nd;
    cands[1].imageId = 2; cands[1].vladDistance = 0.20; cands[1].keypoints = wk; cands[1].descriptors = wd;
    std::vector<RankedResult> ranked = verifier.rerank(qk, qd, cands);

    bool ok = same.validModel && !diff.validModel && same.numInliers > 5 * std::max(1, diff.numInliers)
              && ranked.front().imageId == 2;

    cv::imwrite("results/selftest_same.jpg", GeometricVerifier::drawResult(img, qk, warped, wk, same));
    cv::imwrite("results/selftest_unrelated.jpg", GeometricVerifier::drawResult(img, qk, noise, nk, diff));
    std::cout << "Re-ranking moved the correct image to rank 1: "
              << (ranked.front().imageId == 2 ? "yes" : "no") << std::endl;
    std::cout << "Saved results/selftest_same.jpg and results/selftest_unrelated.jpg" << std::endl;
    std::cout << (ok ? "SELFTEST PASSED" : "SELFTEST FAILED") << std::endl;
    return ok ? 0 : 1;
}

// ---------------- pair ----------------
int runPair(const std::string& qPath, const std::string& cPath, const std::map<std::string, std::string>& opts) {
    cv::Mat q = loadResized(qPath), c = loadResized(cPath);
    if (q.empty() || c.empty()) { std::cerr << "Could not read one of the images" << std::endl; return 1; }

    std::vector<cv::KeyPoint> qk, ck;
    cv::Mat qd, cd;
    extract(q, qk, qd);
    extract(c, ck, cd);
    std::cout << "Keypoints  query: " << qk.size() << "  candidate: " << ck.size() << std::endl;

    GeometricVerifier verifier = makeVerifier(opts);
    VerificationResult r = verifier.verify(qk, qd, ck, cd);
    printResult("pair", r);

    std::string out = opts.count("out") ? opts.at("out") : "results/matches.jpg";
    cv::imwrite(out, GeometricVerifier::drawResult(q, qk, c, ck, r));
    std::cout << "Saved " << out << std::endl;
    return 0;
}

// ---------------- rerank ----------------
int runRerank(int argc, char** argv) {
    cv::Mat q = loadResized(argv[2]);
    if (q.empty()) { std::cerr << "Could not read " << argv[2] << std::endl; return 1; }
    std::vector<cv::KeyPoint> qk;
    cv::Mat qd;
    extract(q, qk, qd);

    std::vector<std::string> names;
    std::vector<Candidate> cands;
    for (int i = 3; i < argc; ++i) {
        cv::Mat c = loadResized(argv[i]);
        if (c.empty()) { std::cerr << "Skipping " << argv[i] << std::endl; continue; }
        Candidate cand;
        cand.imageId = static_cast<int>(names.size());
        cand.vladDistance = static_cast<double>(names.size());  // stand-in for VLAD order
        extract(c, cand.keypoints, cand.descriptors);
        names.push_back(argv[i]);
        cands.push_back(cand);
    }

    GeometricVerifier verifier;
    std::vector<RankedResult> ranked = verifier.rerank(qk, qd, cands);

    std::cout << "\nBefore (VLAD order):" << std::endl;
    for (size_t i = 0; i < names.size(); ++i) std::cout << "  " << i + 1 << ". " << names[i] << std::endl;
    std::cout << "\nAfter geometric re-ranking:" << std::endl;
    for (size_t i = 0; i < ranked.size(); ++i) {
        std::cout << "  " << i + 1 << ". " << std::left << std::setw(40) << names[ranked[i].imageId]
                  << " inliers: " << ranked[i].verification.numInliers
                  << (ranked[i].verification.validModel ? "" : " (rejected)") << std::endl;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "selftest") return runSelfTest(argv[2]);
    if (argc >= 4 && std::string(argv[1]) == "pair") return runPair(argv[2], argv[3], parseOptions(argc, argv, 4));
    if (argc >= 4 && std::string(argv[1]) == "rerank") return runRerank(argc, argv);

    std::cout << "Usage:\n"
              << "  verify_pair selftest <image>\n"
              << "  verify_pair pair <query> <candidate> [--model H|F] [--ratio 0.75] [--thresh 5] [--out file.jpg]\n"
              << "  verify_pair rerank <query> <cand1> <cand2> ...\n";
    return 0;
}
