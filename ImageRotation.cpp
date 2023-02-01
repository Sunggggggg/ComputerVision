#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include"Flag.h"
#define PI 3.141592

using namespace cv;

Mat ImgRotation(Mat imgGray, int flag) {
	int x, y;
	float deg, pos_x, pos_y;
	int r_x, r_y, dx, dy;
	uchar p1, p2, p3, p4;

	int height = imgGray.rows;
	int width = imgGray.cols;

	/*printf("Input deg : ");
	scanf("%f", &deg);*/

	deg = 45;

	float rad = deg * PI / 180;
	float R[2][2] = {
		{cos(rad),sin(rad)},
		{-sin(rad),cos(rad)}
	};
	
	Mat imgRotate(height, width, CV_8UC1);

	// Move Center and Rota
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			// Move Center
			// pos = rotation pos(Original)
			
			pos_x = R[0][0] * (x - width / 2) + R[0][1] * (y - height / 2);
			pos_y = R[1][0] * (x - width / 2) + R[1][1] * (y - height / 2);

			pos_x += width / 2;
			pos_y += height / 2;

			// pos 값을 수치화 해야함. -> r
			switch (flag)
			{
			case NN:

				r_x = (int)(pos_x + 0.5);
				r_y = (int)(pos_y + 0.5);

				if (r_x >= 0 && r_x < width && r_y >= 0 && r_y < height)
					imgRotate.at<uchar>(y, x) = imgGray.at<uchar>(r_y, r_x);
				else
					imgRotate.at<uchar>(y, x) = 0;
				break;
			case AVG:
				r_x = (int)pos_x;
				r_y = (int)pos_y;

				if (r_x >= 0 && r_x < width - 1 && r_y >= 0 && r_y < height - 1) {
					p1 = imgGray.at<uchar>(r_y, r_x);
					p2 = imgGray.at<uchar>(r_y, r_x + 1);
					p3 = imgGray.at<uchar>(r_y + 1, r_x);
					p4 = imgGray.at<uchar>(r_y + 1, r_x + 1);

					imgRotate.at<uchar>(y, x) = 0.25 * (p1 + p2 + p3 + p4);
				}
				else
					imgRotate.at<uchar>(y, x) = 0;
				break;
			case BL:
				r_x = (int)pos_x;
				r_y = (int)pos_y;

				dx = pos_x - r_x;
				dy = pos_y - r_y;

				if (r_x >= 0 && r_x < width - 1 && r_y >= 0 && r_y < height - 1) {
					p1 = imgGray.at<uchar>(r_y, r_x);
					p2 = imgGray.at<uchar>(r_y, r_x + 1);
					p3 = imgGray.at<uchar>(r_y + 1, r_x);
					p4 = imgGray.at<uchar>(r_y + 1, r_x + 1);

					imgRotate.at<uchar>(y, x) =
						p1 * (1 - dx) * (1 - dy)
						+ p2 * dx * (1 - dy)
						+ p3 * (1 - dx) * dy
						+ p4 * dx * dy;
				}
				else
					imgRotate.at<uchar>(y, x) = 0;
				break;
			default:
				break;
			}
		}
	}

	return imgRotate;
}