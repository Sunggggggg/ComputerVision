#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include"Flag.h"

using namespace cv;

Mat ImgResize(Mat imgGray, int flag) {
	int x, y;
	float scale, pos_x, pos_y;
	int r_x, r_y;
	float dx, dy;
	uchar p1, p2, p3, p4;
	
	printf("Input Scale : ");
	scanf("%f", &scale);

	int re_height = int(imgGray.rows * scale);
	int re_width = int(imgGray.cols * scale);

	Mat imgResize(re_height, re_width, CV_8UC1);

	// Backward
	for (y = 0; y < re_height; y++) {
		for (x = 0; x < re_width; x++) {
			pos_x = (1.0 / scale) * x;
			pos_y = (1.0 / scale) * y;
			
			switch (flag)
			{
			case NN:
				// 1.1이면 1에 가깝고, 1.6이면 2에 가깝다는 것을 표현
				r_x = (int)(pos_x + 0.5);
				r_y = (int)(pos_y + 0.5);

				imgResize.at<uchar>(y, x) = imgGray.at<uchar>(r_y, r_x);
				
				break;
			case AVG:
				// Av
				r_x = (int)pos_x;
				r_y = (int)pos_y;

				p1 = imgGray.at<uchar>(r_y, r_x);
				p2 = imgGray.at<uchar>(r_y, r_x + 1);
				p3 = imgGray.at<uchar>(r_y + 1, r_x);
				p4 = imgGray.at<uchar>(r_y + 1, r_x + 1);

				imgResize.at<uchar>(y, x) = uchar(0.25 * (p1 + p2 + p3 + p4));

				break;

			case BL:
				r_x = (int)pos_x;
				r_y = (int)pos_y;

				dx = (float)pos_x - (float)r_x;
				dy = (float)pos_y - (float)r_y;

				p1 = imgGray.at<uchar>(r_y, r_x);
				p2 = imgGray.at<uchar>(r_y, r_x + 1);
				p3 = imgGray.at<uchar>(r_y + 1, r_x);
				p4 = imgGray.at<uchar>(r_y + 1, r_x + 1);

				imgResize.at<uchar>(y, x) =
					(p1 * (1 - dx) * (1 - dy) + p2 * dx * (1 - dy)
						+ p3 * (1 - dx) * dy + p4 * dx * dy);

				break;
			default:
				break;
			}

		}
	}

	return imgResize;
}


Mat ImgResize_Forward(Mat imgGray) {
	int x, y, height, width, re_height, re_width;
	float scale, pos_x, pos_y;
	int r_x, r_y;

	printf("Input Scale : ");
	scanf("%f", &scale);

	height = imgGray.rows;
	width = imgGray.cols;

	re_height = int(height * scale);
	re_width = int(width * scale);

	Mat imgResize(re_height, re_width, CV_8UC1);

	// Forward filling (NN) 방법을 사용
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			pos_x = x * scale;
			pos_y = y * scale;

			r_x = int(pos_x + 0.5);
			r_y = int(pos_y + 0.5);

			imgResize.at<uchar>(r_y, r_x) = imgGray.at<uchar>(y, x);
		}
	}
	return imgResize;
}