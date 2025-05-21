#include <opencv2/opencv.hpp>
#include "Resizing.h"
#include <iostream>


using namespace cv;
using namespace std;

double getPSNR(Mat I1, Mat I2) {
    Mat s1;
    absdiff(I1, I2, s1);
    s1.convertTo(s1, CV_32F);
    s1 = s1.mul(s1);
    Scalar s = sum(s1);
    double sse = s[0] + s[1] + s[2];
    if (sse <= 1e-10) return INFINITY;
    double mse = sse / (double)(I1.channels() * I1.total());
    return 10.0 * log10((255 * 255) / mse);
}

int psnrToScore(double psnr) {
    if (psnr == INFINITY) return 100;
    if (psnr < 10) return 0;
    if (psnr > 40) return 100;
    return static_cast<int>((psnr - 10.0) * 100.0 / 30.0);
}

void compareAndShow(const string& label, const Mat& custom, const Mat& reference) {
    if (!custom.empty() && !reference.empty() && custom.size() == reference.size()) {
        double psnr = getPSNR(custom, reference);
        int score = psnrToScore(psnr);
        cout << label << ": "  << psnr << " dB (" << score << "% similaritate)" << "\n";
    } else {
        cout << label << ": [comparatie imposibila]" << "\n";
    }
}

int main() {
    int imgNumber;
    cout << "Introdu numarul imaginii de test (1 - 8):";
    cin >> imgNumber;

    if (imgNumber < 1 || imgNumber > 8) {
        cout << "Numar invalid. Trebuie sa fie intre 1 si 8.\n";
        return -1;
    }

    string imagePath = "C:/Users/Monica/Desktop/an 3/ProiectPI1/img" + to_string(imgNumber) + ".bmp";
    Mat image = imread(imagePath, IMREAD_COLOR);

    if (image.empty()) {
        cout << "Eroare la citirea imaginii." << "\n";
        return -1;
    }

    double scale_up = 2.0;
    double scale_down = 0.5;

    string metoda, directie;
    cout << "Introdu metoda dorita (NN, BILINEAR, BICUBIC, PIXEL, AREA):";
    cin >> metoda;

    if (metoda == "NN" || metoda == "BILINEAR" || metoda == "BICUBIC") {
        cout << "Introdu directia (UP, DOWN, BOTH):";
        cin >> directie;
    }
    cout << "\n";

    if (metoda == "NN") {
        if (directie == "UP" || directie == "BOTH") {
            Mat custom = resizeNearestNeighbor(image, scale_up);
            Mat ocv;
            resize(image, ocv, Size(), scale_up, scale_up, INTER_NEAREST);
            compareAndShow("NN Up", custom, ocv);
            imshow("NN Custom Up", custom);
            imshow("OpenCV NN Up", ocv);
        }
        if (directie == "DOWN" || directie == "BOTH") {
            Mat custom = resizeNearestNeighbor(image, scale_down);
            Mat ocv;
            resize(image, ocv, Size(), scale_down, scale_down, INTER_NEAREST);
            resize(custom, custom, ocv.size());
            compareAndShow("NN Down", custom, ocv);
            imshow("NN Custom Down", custom);
            imshow("OpenCV NN Down", ocv);
        }
    }

    else if (metoda == "BILINEAR") {
        if (directie == "UP" || directie == "BOTH") {
            Mat custom = resizeBilinear(image, scale_up);
            Mat ocv;
            resize(image, ocv, Size(), scale_up, scale_up, INTER_LINEAR);
            compareAndShow("Bilinear Up", custom, ocv);
            imshow("Custom Bilinear Up", custom);
            imshow("OpenCV Bilinear Up", ocv);
        }
        if (directie == "DOWN" || directie == "BOTH") {
            Mat custom = resizeBilinear(image, scale_down);
            Mat ocv;
            resize(image, ocv, Size(), scale_down, scale_down, INTER_LINEAR);
            resize(custom, custom, ocv.size());
            compareAndShow("Bilinear Down", custom, ocv);
            imshow("Custom Bilinear Down", custom);
            imshow("OpenCV Bilinear Down", ocv);
        }
    }

    else if (metoda == "BICUBIC") {
        if (directie == "UP" || directie == "BOTH") {
            Mat custom = resizeBicubic(image, scale_up);
            Mat ocv;
            resize(image, ocv, Size(), scale_up, scale_up, INTER_CUBIC);
            compareAndShow("Bicubic Up", custom, ocv);
            imshow("Custom Bicubic Up", custom);
            imshow("OpenCV Bicubic Up", ocv);
        }
        if (directie == "DOWN" || directie == "BOTH") {
            Mat custom = resizeBicubic(image, scale_down);
            Mat ocv;
            resize(image, ocv, Size(), scale_down, scale_down, INTER_CUBIC);
            resize(custom, custom, ocv.size());
            compareAndShow("Bicubic Down", custom, ocv);
            imshow("Custom Bicubic Down", custom);
            imshow("OpenCV Bicubic Down", ocv);
        }
    }

    else if (metoda == "AREA") {
            Mat custom = resizeAreaAverage(image, scale_down);
            Mat ocv;
            resize(image, ocv, Size(), scale_down, scale_down, INTER_AREA);
            resize(custom, custom, ocv.size());
            compareAndShow("Area Average Down", custom, ocv);
            imshow("Custom Area Down", custom);
            imshow("OpenCV Area Down", ocv);

    }

    else if (metoda == "PIXEL") {
        Mat custom = pixelReplication(image, static_cast<int>(scale_up));
        imshow("Custom Pixel Replication", custom);
        cout << "Pixel Replication: [fara echivalent OpenCV]" << "\n";
    }

    else {
        cout << "Metoda necunoscuta!" << "\n";
    }

    waitKey(0);

    return 0;
}
