#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include"SizeDefine.h"

using namespace cv;

void FilterCreate(float a[][Kernel_SIZE]) {
	int x, y;
	float sigma = 1.0;
	float d, s = 2.0 * sigma * sigma;

	for (y = -Kernel_SIZE / 2; y <= Kernel_SIZE / 2; y++) {
		for (x = -Kernel_SIZE / 2; x <= Kernel_SIZE / 2; x++) {
			d = sqrt(x * x + y * y);
			a[y + Kernel_SIZE / 2][x + Kernel_SIZE / 2]
				= exp(-(d * d) / s) / (sqrt(2 * PI) * sigma);
		}
	}

}

Mat GetGaussianFilter(Mat img){
	int x, y, xx, yy, height, width;
	float sum, min = 100000, max = -1;
	
	height = img.rows;
	width = img.cols;

	float GKernel[Kernel_SIZE][Kernel_SIZE];
	FilterCreate(GKernel);

	float* mag = (float*)calloc(height * width, sizeof(float));
	Mat imgBlur(height, width, CV_8UC1);

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			sum = 0;

			for (yy = y - Kernel_SIZE / 2; yy <= y + Kernel_SIZE / 2; yy++) {
				for (xx = x - Kernel_SIZE / 2; xx <= x + Kernel_SIZE / 2; xx++) {
					if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
						sum += img.at<uchar>(yy,xx) * GKernel[yy - (y - Kernel_SIZE / 2)][xx - (x - Kernel_SIZE / 2)];
					}
				}
			}
			mag[y * width + x] = sum;
			if (max < mag[y * width + x]) max = mag[y * width + x];
			if (min > mag[y * width + x]) min = mag[y * width + x];
		}
	}

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			imgBlur.at<uchar>(y, x) = (255 - 0) * (mag[y * width + x] - min) / (max - min) + 0;
		}
	}

	free(mag);
	return imgBlur;
}