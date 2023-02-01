#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include"SizeDefine.h"
#define PI 3.141592

using namespace cv;

float** GetHistogram(Mat img, int BLK, int BIN) {
	int i, j, x, y, xx, yy, height, width, idx, k = -1;
	int t;

	float sumx, sumy, mag, deg, rad;
	float min = 100000, max = -1;

	height = img.rows;
	width = img.cols;

	int mask_x[MSIZE * MSIZE] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
	int mask_y[MSIZE * MSIZE] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

	int Histo_row = ((height - BLK) / (BLK / 2) + 1) * ((width - BLK) / (BLK / 2) + 1);
	float** HOG = (float**)calloc(Histo_row, sizeof(float*));

	for (j = 0; j <= height - BLK; j += BLK / 2) {
		for (i = 0; i <= width - BLK; i += BLK / 2) {
			k++;
			*(HOG + k) = (float*)calloc(BIN, sizeof(float));

			for (y = j; y < j + BLK; y++) {
				for (x = i; x < i + BLK; x++) {
					sumx = 0.0;
					sumy = 0.0;
					
					for (yy = y - MSIZE / 2; yy <= y + MSIZE / 2; yy++) {
						for (xx = x - MSIZE / 2; xx <= x + MSIZE / 2; xx++) {
							if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
								sumx += img.at<uchar>(yy, xx) * mask_x[(yy - (y - MSIZE / 2)) * MSIZE + (xx - (x - MSIZE / 2))];
								sumy += img.at<uchar>(yy, xx) * mask_y[(yy - (y - MSIZE / 2)) * MSIZE + (xx - (x - MSIZE / 2))];
							}
						}
					}
					sumx /= MSIZE * MSIZE;
					sumy /= MSIZE * MSIZE;

					mag = sqrt(sumx * sumx + sumy * sumy);

					rad = atan2(sumy, sumx);
					deg = rad * 180 / PI;
					if (deg < 0) deg += 180;
					idx = deg / 20.0;
					
					HOG[k][idx] += mag;

				}
			}
		}
	}

	float sum;
	for (y = 0; y < Histo_row; y++) {
		sum = 0.0;
		for (x = 0; x < BIN; x++)
			sum += HOG[y][x] * HOG[y][x];
		for (x = 0; x < BIN; x++)
			HOG[y][x] /= sqrt(sum);
	}
	
	return HOG;
}

float CompareHistogram(float** a, float** b, int height, int width) {
	int i, j;
	float score = 0.0;
	for (j = 0; j < height; j++) {
		for (i = 0; i < width; i++) {
			score += abs(a[j][i] - b[j][i]);
		}
	}
	//printf("%f\n", score);
	return score;
}

void freePtr(float** a, int row) {
	int j;
	for (j = 0; j < row; j++) free(*(a + j));
	free(a);
}



void printHOG(float** a, int row, int col) {
	int i, j;

	for (j = 0; j < row; j++) {
		for (i = 0; i < col; i++) {
			printf("%3.2f ", a[j][i]);
		}
		printf("\n");
	}
}



void Savecsv(float** a, int height, int width) {
	int i, j;
	char buffer[256];
	FILE* fp;
	fp = fopen("Histogram.csv", "w");

	for (j = 0; j < height; j++) {
		for (i = 0; i < width; i++) {
			if (i != 8) {
				sprintf(buffer, "%f, ", a[j][i]);
				fprintf(fp, buffer);
			}
			else {
				sprintf(buffer, "%f\n", a[j][i]);
				fprintf(fp, buffer);
			}
		}
	}

	fclose(fp);
}