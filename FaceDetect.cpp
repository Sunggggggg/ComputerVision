#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

#include"Flag.h"
#include"SizeDefine.h"

using namespace cv;
using namespace cv::ml;
using namespace std;

void freePtr(float** a, int row);
float CompareHistogram(float** a, float** b, int height, int width);
float** GetHistogram(Mat img, int BLK, int BIN);

void GetFaceDetect(string filename) {
	int x, y;
	CascadeClassifier cascade;
	cascade.load("C:/Users/sky/OpenCV_Version/opencv_345/sources/data/lbpcascades/lbpcascade_frontalface.xml");

	Mat img = imread(filename, 1);

	vector<Rect> faces;
	cascade.detectMultiScale(img, faces, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));

	for (y = 0; y < faces.size(); y++) {
		Point lb(faces[y].x + faces[y].width, faces[y].y + faces[y].height);
		Point tr(faces[y].x, faces[y].y);
		rectangle(img, lb, tr, Scalar(0, 255, 0), 3, 8, 0);
	}

	imshow("Face", img);
	waitKey(0);
}

float GetSimilarity(float** ref, float** tar, int Hrow, int bin, int flag) {
	int x, y;
	float corelation = 0.0;

	switch (flag) {
	case COSINE:
		float refMag, tarMag, Pro;

		refMag = 0.0;
		tarMag = 0.0;
		Pro = 0.0;

		for (y = 0; y < Hrow; y++) {
			for (x = 0; x < bin; x++) {
				refMag += ref[y][x] * ref[y][x];
				tarMag += tar[y][x] * tar[y][x];
				Pro += ref[y][x] * tar[y][x];
			}
		}
		corelation = Pro / (sqrt(refMag) * sqrt(tarMag));
		break;

	case EU:
		corelation = CompareHistogram(ref, tar, Hrow, bin);
		break;

	default:
		corelation = 0.0;
		break;
	}

	return corelation;
}

float ThCosine = 0.68, ThEu = 200;

Mat GeneralGetFaceDetect(Mat imgRef, Mat imgTar) {
	// 1. ref HOG 
	int blk, bin, HrowRef, heightRef, widthRef;
	float** HOGRef;
	blk = 6;
	bin = 9;
	heightRef = imgRef.rows;
	widthRef = imgRef.cols;

	HrowRef = ((heightRef - blk) / (blk / 2) + 1)
		* ((widthRef - blk) / (blk / 2) + 1);

	HOGRef = GetHistogram(imgRef, blk, bin);

	// 2. tar HOG
	int x1, x2, y1, y2, heightTar, widthTar;
	float corelation;

	heightTar = imgTar.rows;
	widthTar = imgTar.cols;

	Mat DetectMap = Mat::zeros(heightTar, widthTar, CV_8UC1);

	for (y1 = 0; y1 < heightTar - heightRef; y1++) {
		for (x1 = 0; x1 < widthTar - widthRef; x1++) {
			Mat imgTemp(heightRef, widthRef, CV_8UC1);
			float** HOGTar;
			for (y2 = y1; y2 < y1 + heightRef; y2++) {
				for (x2 = x1; x2 < x1 + widthRef; x2++) {
					imgTemp.at<uchar>(y2 - y1, x2 - x1) = imgTar.at<uchar>(y2, x2);
				}
			}
			HOGTar = GetHistogram(imgTemp, blk, bin);

			// 3. Similarity
			corelation = GetSimilarity(HOGRef, HOGTar, HrowRef, bin, EU);
			//printf("x : %d y : %d corelation : %f\n", x1 + widthRef / 2, y1 + heightRef / 2, corelation);
			if (corelation < ThEu) {
				DetectMap.at<uchar>(y1 + heightRef / 2, x1 + widthRef / 2) = (uchar)255;
			}
			freePtr(HOGTar, HrowRef);
		}
	}

	// 4. Display Result
	cvtColor(imgTar, imgTar, COLOR_GRAY2BGR);
	Point center1, center2;
	Scalar green = Scalar(0, 255, 0);
	int r = 10;

	for (y1 = 0; y1 < heightTar; y1++) {
		for (x1 = 0; x1 < widthTar; x1++) {
			if (DetectMap.at<uchar>(y1, x1)) {
				//circle(imgTar, Point(x1, y1), r, green);

				center1.x = x1 - widthRef / 2;
				center1.y = y1 - heightRef / 2;
				center2.x = x1 + widthRef / 2;
				center2.y = y1 + heightRef / 2;
				rectangle(imgTar, center1, center2, green);
			}

		}
	}

	freePtr(HOGRef, HrowRef);

	return imgTar;
}