#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#define SIZE 3
#define PI 3.141592

using namespace cv;

int fx[SIZE][SIZE] = {
	{-1,0,1},
	{-1,0,1},
	{-1,0,1}
};

int fy[SIZE][SIZE] = {
	{-1,-1,-1},
	{0,0,0},
	{1,1,1}
};

Mat EdgeMagnitude(Mat imgGray) {
	int i, j, x, y, convx, convy;
	
	int height = imgGray.rows;
	int width = imgGray.cols;

	Mat imgEdge(height, width, CV_8UC1);

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			convx = 0;
			convy = 0;

			// inner product
			for (i = 0; i < SIZE; i++) {
				for (j = 0; j < SIZE; j++) {

					if (y - SIZE / 2 + i < 0 || y - SIZE / 2 + i >= height)
						convy = 0;
					else if (x - SIZE / 2 + j < 0 || x - SIZE / 2 + j >= width)
						convx = 0;
					else {
						convx += imgGray.at<uchar>(y - SIZE / 2 + i, x - SIZE / 2 + j) * fx[i][j];
						convy += imgGray.at<uchar>(y - SIZE / 2 + i, x - SIZE / 2 + j) * fy[i][j];
					}
				}
			}
			imgEdge.at<uchar>(y, x) = uchar(sqrt(convx * convx + convy * convy));
		}
	}

	return imgEdge;
}


Mat EdgeMagnitude_Extra(Mat imgGray) {
	int i, j, x, y;

	int height = imgGray.rows;
	int width = imgGray.cols;

	float convx, convy, deg,rad;
	float min = 100000, max = -1;
	float* mag = (float*)calloc(height * width, sizeof(float));
	int* histogram = (int*)calloc(9, sizeof(int));

	int mask_x[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
	int mask_y[9] = { - 1, -1, -1, 0, 0, 0, 1, 1, 1};
	

	Mat imgEdge(height, width, CV_8UC1);

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			convx = 0;
			convy = 0;

			// inner product
			for (i = y - 1; i <= y+1; i++) {
				for (j = x-1; j <= x+1; j++) {
					if (i >= 0 && i < height && j >= 0 && j < width) {
						convx += imgGray.at<uchar>(i, j) * mask_x[(i - (y - 1)) * 3 + (j - (x - 1))];
						convy += imgGray.at<uchar>(i, j) * mask_y[(i - (y - 1)) * 3 + (j - (x - 1))];
					}
				}
			}
			convx /= 9.0;
			convy /= 9.0;

			mag[y * width + x] = sqrt(convx * convx + convy * convy);
			if (max < mag[y * width + x]) max = mag[y * width + x];
			if (min > mag[y * width + x]) min = mag[y * width + x];

			// dir
			rad = atan2(convy, convx);
			deg = rad * 180 / PI;
			if (deg < 0) deg += 180;

			int idx = deg / 20.0;
			histogram[idx] += mag[y * width + x];
		}

	}


	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			imgEdge.at<uchar>(y, x) = 255 * (mag[y * width + x] - min) / (max - min);
		}
	}

	free(mag);
	free(histogram);

	return imgEdge;
}