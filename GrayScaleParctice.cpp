#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>

#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>

#define BLK 30
using namespace cv;

Mat MakeGrayPart (Mat img) {
	int i, j;
	int rVal, gVal, bVal;
	int cx, cy;
	
	printf("Center x : ");
	scanf("%d", &cx);

	printf("Center y : ");
	scanf("%d", &cy);

	for (i = cx - BLK; i <= cx + BLK; i++) {
		for (j = cy - BLK; j <= cy + BLK; j++) {
			rVal = img.at<Vec3b>(j, i)[2];
			gVal = img.at<Vec3b>(j, i)[1];
			bVal = img.at<Vec3b>(j, i)[0];

			img.at<Vec3b>(j, i)[2] = (rVal + gVal + bVal) / 3;
			img.at<Vec3b>(j, i)[1] = (rVal + gVal + bVal) / 3;
			img.at<Vec3b>(j, i)[0] = (rVal + gVal + bVal) / 3;
		}
	}
	return img;
}