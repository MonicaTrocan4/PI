#include "Resizing.h"


Mat resizeNearestNeighbor(Mat src, double scale) {

    int newRows = static_cast<int>(src.rows * scale);
    int newCols = static_cast<int>(src.cols * scale);
    Mat dst(newRows, newCols, src.type());

    for (int i = 0; i < newRows; i++) {
        for (int j = 0; j < newCols; j++) {
            int srcY = min(static_cast<int>(i / scale), src.rows - 1);
            int srcX = min(static_cast<int>(j / scale), src.cols - 1);

            if (src.channels() == 1) {
                dst.at<uchar>(i, j) = src.at<uchar>(srcY, srcX);
            } else if (src.channels() == 3) {
                dst.at<Vec3b>(i, j) = src.at<Vec3b>(srcY, srcX);
            }
        }
    }

    return dst;
}

Mat resizeBilinear(Mat src, double scale) {
    
    int newRows = static_cast<int>(src.rows * scale);
    int newCols = static_cast<int>(src.cols * scale);
    Mat dst(newRows, newCols, src.type());

    for (int i = 0; i < newRows; i++) {
        for (int j = 0; j < newCols; j++) {
            float gx = j / scale;
            float gy = i / scale;

            int x0 = static_cast<int>(gx);
            int y0 = static_cast<int>(gy);
            int x1 = min(x0 + 1, src.cols - 1);
            int y1 = min(y0 + 1, src.rows - 1);

            float a = gx - x0;
            float b = gy - y0;

            if (src.channels() == 1) {
                // Grayscale
                float p0 = src.at<uchar>(y0, x0);
                float p1 = src.at<uchar>(y0, x1);
                float p2 = src.at<uchar>(y1, x0);
                float p3 = src.at<uchar>(y1, x1);

                float value = (1 - a) * (1 - b) * p0 +
                              a * (1 - b) * p1 +
                              (1 - a) * b * p2 +
                              a * b * p3;

                dst.at<uchar>(i, j) = static_cast<uchar>(value);
            } else if (src.channels() == 3) {
                // Color
                Vec3b p0 = src.at<Vec3b>(y0, x0);
                Vec3b p1 = src.at<Vec3b>(y0, x1);
                Vec3b p2 = src.at<Vec3b>(y1, x0);
                Vec3b p3 = src.at<Vec3b>(y1, x1);

                Vec3b value;
                for (int c = 0; c < 3; ++c) {
                    value[c] = static_cast<uchar>(
                        (1 - a) * (1 - b) * p0[c] +
                        a * (1 - b) * p1[c] +
                        (1 - a) * b * p2[c] +
                        a * b * p3[c]
                    );
                }

                dst.at<Vec3b>(i, j) = value;
            }
        }
    }

    return dst;
}


float cubicInterpolate(float v0, float v1, float v2, float v3, float x) {
    return v1 + 0.5f * x * (v2 - v0 + x * (2.0f * v0 - 5.0f * v1 + 4.0f * v2 - v3 + x * (3.0f * (v1 - v2) + v3 - v0)));
}

Mat resizeBicubic(Mat src, double scale) {
    int newRows = static_cast<int>(src.rows * scale);
    int newCols = static_cast<int>(src.cols * scale);
    Mat dst(newRows, newCols, src.type());

    for (int y = 0; y < newRows; ++y) {
        for (int x = 0; x < newCols; ++x) {
            float gx = x / scale;
            float gy = y / scale;
            int x1 = static_cast<int>(gx);
            int y1 = static_cast<int>(gy);
            float dx = gx - x1;
            float dy = gy - y1;

            if (src.channels() == 1) {
                float patch[4][4];
                for (int m = -1; m <= 2; ++m) {
                    for (int n = -1; n <= 2; ++n) {
                        int px = max(0, min(src.cols - 1, x1 + n));
                        int py = max(0, min(src.rows - 1, y1 + m));
                        patch[m + 1][n + 1] = static_cast<float>(src.at<uchar>(py, px));
                    }
                }

                float col[4];
                for (int i = 0; i < 4; ++i)
                    col[i] = cubicInterpolate(patch[i][0], patch[i][1], patch[i][2], patch[i][3], dx);

                float value = cubicInterpolate(col[0], col[1], col[2], col[3], dy);
                value = max(0.f, min(255.f, value));
                dst.at<uchar>(y, x) = static_cast<uchar>(value);

            } else if (src.channels() == 3) {
                Vec3f patch[4][4];
                for (int m = -1; m <= 2; ++m) {
                    for (int n = -1; n <= 2; ++n) {
                        int px = max(0, min(src.cols - 1, x1 + n));
                        int py = max(0, min(src.rows - 1, y1 + m));
                        Vec3b p = src.at<Vec3b>(py, px);
                        patch[m + 1][n + 1] = Vec3f(p[0], p[1], p[2]);
                    }
                }

                Vec3f col[4];
                for (int i = 0; i < 4; ++i)
                    for (int c = 0; c < 3; ++c)
                        col[i][c] = cubicInterpolate(
                            patch[i][0][c], patch[i][1][c], patch[i][2][c], patch[i][3][c], dx
                        );

                Vec3b result;
                for (int c = 0; c < 3; ++c) {
                    float val = cubicInterpolate(col[0][c], col[1][c], col[2][c], col[3][c], dy);
                    val = max(0.f, min(255.f, val));
                    result[c] = static_cast<uchar>(val);
                }

                dst.at<Vec3b>(y, x) = result;
            }
        }
    }

    return dst;
}

Mat pixelReplication(Mat src, int factor) {
    int newRows = src.rows * factor;
    int newCols = src.cols * factor;
    Mat dst(newRows, newCols, src.type());

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            if (src.channels() == 1) {
                uchar value = src.at<uchar>(y, x);
                for (int i = 0; i < factor; ++i) {
                    for (int j = 0; j < factor; ++j) {
                        dst.at<uchar>(y * factor + i, x * factor + j) = value;
                    }
                }
            } else if (src.channels() == 3) {
                Vec3b value = src.at<Vec3b>(y, x);
                for (int i = 0; i < factor; ++i) {
                    for (int j = 0; j < factor; ++j) {
                        dst.at<Vec3b>(y * factor + i, x * factor + j) = value;
                    }
                }
            }
        }
    }

    return dst;
}

Mat resizeAreaAverage(Mat src, double scale) {
    int newRows = static_cast<int>(src.rows * scale);
    int newCols = static_cast<int>(src.cols * scale);
    Mat dst(newRows, newCols, src.type());

    int blockHeight = static_cast<int>(1.0 / scale);
    int blockWidth = static_cast<int>(1.0 / scale);

    for (int y = 0; y < newRows; ++y) {
        for (int x = 0; x < newCols; ++x) {
            int startY = y * blockHeight;
            int startX = x * blockWidth;
            int endY = min(startY + blockHeight, src.rows);
            int endX = min(startX + blockWidth, src.cols);

            if (src.channels() == 1) {
                int sum = 0;
                int count = 0;
                for (int i = startY; i < endY; ++i) {
                    for (int j = startX; j < endX; ++j) {
                        sum += src.at<uchar>(i, j);
                        ++count;
                    }
                }
                dst.at<uchar>(y, x) = static_cast<uchar>(sum / count);
            } else if (src.channels() == 3) {
                Vec3i sum = Vec3i(0, 0, 0);
                int count = 0;
                for (int i = startY; i < endY; ++i) {
                    for (int j = startX; j < endX; ++j) {
                        Vec3b pixel = src.at<Vec3b>(i, j);
                        sum[0] += pixel[0];
                        sum[1] += pixel[1];
                        sum[2] += pixel[2];
                        ++count;
                    }
                }
                Vec3b avg;
                avg[0] = static_cast<uchar>(sum[0] / count);
                avg[1] = static_cast<uchar>(sum[1] / count);
                avg[2] = static_cast<uchar>(sum[2] / count);
                dst.at<Vec3b>(y, x) = avg;
            }
        }
    }

    return dst;
}


