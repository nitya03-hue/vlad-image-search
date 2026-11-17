#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main()
{
    // Step 1: Load the test image.
    Mat image = imread("data/sample/test.jpg");

    if (image.empty()) {
        cout << "Error: Could not load test.jpg" << endl;
        return 1;
    }

    // Step 2: Convert the image to grayscale.
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Step 3: Detect SIFT keypoints and descriptors.
    Ptr<SIFT> sift = SIFT::create();

    vector<KeyPoint> keypoints;
    Mat descriptors;

    sift->detectAndCompute(gray, noArray(),
                           keypoints, descriptors);

    // Step 4: Print feature information.
    cout << "SIFT Feature Extraction" << endl;
    cout << "Number of keypoints: "
         << keypoints.size() << endl;

    cout << "Descriptor matrix: "
         << descriptors.rows << " x "
         << descriptors.cols << endl;

    // Step 5: Draw detected keypoints.
    Mat output;
    drawKeypoints(image, keypoints, output,
                  Scalar(0, 255, 0));

    // Step 6: Save the output image.
    bool saved = imwrite("results/sift_keypoints.jpg", output);

    if (!saved) {
        cout << "Error: Could not save output image." << endl;
        return 1;
    }

    cout << "Output saved to results/sift_keypoints.jpg"
         << endl;

    return 0;
}
