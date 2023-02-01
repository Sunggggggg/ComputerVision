#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <math.h>
#include <opencv2/photo.hpp>
using namespace std;
using namespace cv;

Mat GetIpainting(Mat imgDoodle) {
    int heigh = imgDoodle.rows;
    int width = imgDoodle.cols;

    Mat imgMask = Mat::zeros(heigh, width, CV_8UC1);
    for (int y = 0; y < heigh; y++) {
        for (int x = 0; x < width; x++) {
            if ((imgDoodle.at<Vec3b>(y, x)[0] == 0)
                && (imgDoodle.at<Vec3b>(y, x)[1] == 255)
                && (imgDoodle.at<Vec3b>(y, x)[2] == 255)) imgMask.at<uchar>(y, x) = 255;

        }
    }
    Mat imgOut;
    inpaint(imgDoodle, imgMask, imgOut, 10, INPAINT_TELEA);
    imshow("imgOut", imgOut);
    waitKey(0);

    return imgOut;
}