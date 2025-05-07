
#include <opencv2/opencv.hpp>
#include "Resizing.h"
#include <iostream>

using namespace cv;
using namespace std;


int main() {
    Mat image = imread("C:/Users/Monica/Desktop/an 3/ProiectPI1/img.bmp", IMREAD_COLOR);

    imshow("Original", image);

    double scale_up = 3.0;
    double scale_down = 0.5;

    Mat nn_up = resizeNearestNeighbor(image, scale_up);
    Mat bilinear_up = resizeBilinear(image, scale_up);
    Mat replication_up = pixelReplication(image, 3);
    Mat bicubic_up = resizeBicubic(image, scale_up);

    Mat nn_down = resizeNearestNeighbor(image, scale_down);
    Mat bilinear_down = resizeBilinear(image, scale_down);
    Mat bicubic_down = resizeBicubic(image, scale_down);
    Mat aa_down = resizeAreaAverage(image, scale_down);

    // imshow("NN Up", nn_up);
    // imshow("Bilinear Up", bilinear_up);
    // imshow("Replication", replication_up);
    // imshow("Bicubic Up", bicubic_up);

     imshow("NN Down", nn_down);
     imshow("Bilinear Down", bilinear_down);
     imshow("Bicubic Down", bicubic_down);
     imshow("Area Average", aa_down);

    waitKey(0);
    return 0;
}
