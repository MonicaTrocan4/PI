
#ifndef RESIZING_H
#define RESIZING_H

#endif //RESIZING_H

#include<opencv2/opencv.hpp>

using namespace cv;

Mat resizeNearestNeighbor(Mat src, double scale);
Mat resizeBilinear(Mat src, double scale);
Mat resizeBicubic(Mat src, double scale);
float cubicInterpolate(float v0, float v1, float v2, float v3, float x);
Mat pixelReplication(Mat src, int factor);
Mat resizeAreaAverage(Mat src, double scale);


