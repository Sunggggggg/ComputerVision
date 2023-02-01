#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include"Flag.h"
#include"SizeDefine.h"

using namespace cv;
using namespace std;

int mask_x[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
int mask_y[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

void freePtr(float** a, int row);

// Only Harris Corner
int* nonMaximaSuppression(float* R, int* c, int height, int width, int wsize) {
	int x, y, xx, yy, max_x = 0, max_y = 0;
	float max;
	int* newCorner = (int*)calloc(height * width, sizeof(int));

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			max = -1;

			if (c[y * width + x]) {
				for (yy = y - (wsize / 2); yy <= y + (wsize / 2); yy++) {
					for (xx = x - (wsize / 2); xx <= x + (wsize / 2); xx++) {
						if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
							if (c[yy * width + xx]) {
								if (max < R[y * width + x]) {
									max = R[y * width + x];
									max_x = xx;
									max_y = yy;
								}
							}
							c[yy * width + xx] = 0;
						}
					}
				}
				// 나머지는 다 0이고 max정보 값들만 갖고 나옴.
				newCorner[max_y * width + max_x] = 1;
			}
		}
	}


	return newCorner;

}
Mat GetHarrisCorner(Mat img, int th, int wsize) {
	int x, y, xx, yy, height, width, sumx, sumy, IxIx, IxIy, IyIx, IyIy, det,tr;
	int max = -1, min = 100000;
	float k = 0.04;
	
	height = img.rows;
	width = img.cols;

	Mat imgCorner(height, width, CV_8UC3);

	int* lx = (int*)calloc(height * width, sizeof(int));
	int* ly = (int*)calloc(height * width, sizeof(int));
	int* corner = (int*)calloc(height * width, sizeof(int));
	
	float* R = (float*)calloc(height * width, sizeof(float));
	
	Scalar c;
	Point pCenter;
	int radius = 1;

	// Compute Gradient And Convert Channel
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			sumx = 0;
			sumy = 0;
			for (yy = y - MSIZE / 2; yy <= y + MSIZE / 2; yy++) {
				for (xx = x - MSIZE / 2; xx <= x + MSIZE / 2; xx++) {
					if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
						sumx += img.at<uchar>(yy, xx) * mask_x[(yy - (y - MSIZE / 2)) * MSIZE + (xx - (x - MSIZE / 2))];
						sumy += img.at<uchar>(yy, xx) * mask_y[(yy - (y - MSIZE / 2)) * MSIZE + (xx - (x - MSIZE / 2))];
					}		
				}
				lx[y * width + x] = sumx / (MSIZE * MSIZE);
				ly[y * width + x] = sumy / (MSIZE * MSIZE);

				imgCorner.at<Vec3b>(y, x)[0] = img.at<uchar>(y, x);
				imgCorner.at<Vec3b>(y, x)[1] = img.at<uchar>(y, x);
				imgCorner.at<Vec3b>(y, x)[2] = img.at<uchar>(y, x);

			}
		}
	}

	// Compute R And Detect Corner 
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			IxIx=0.0;
			IxIy=0.0;
			IyIx=0.0;
			IyIy=0.0;
			for (yy = y - (wsize / 2); yy <= y + (wsize / 2); yy++) {
				for (xx = x - (wsize / 2); xx <= x + (wsize / 2); xx++) {
					if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
						IxIx += lx[yy * width + xx] * lx[yy * width + xx];
						IxIy += lx[yy * width + xx] * ly[yy * width + xx];
						IyIx += ly[yy * width + xx] * lx[yy * width + xx];
						IyIy += ly[yy * width + xx] * ly[yy * width + xx];
					}
				}
			}
			det = IxIx * IyIy - IxIy * IyIx;
			tr = IxIx + IyIy;
			R[y * width + x] = det - k * tr * tr;

			if (max < R[y * width + x]) max = R[y * width + x];
			if (min > R[y * width + x]) min = R[y * width + x];
		}
	}

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			R[y * width + x] = 255 * (R[y * width + x] - min) / (max - min);

			if (R[y * width + x] > th)	corner[y * width + x] = 1;
		}
	}

	int t = 0;
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			if (corner[y * width + x]) {
				t++;
			}
		}
	}
	//printf("Corner (Before Suppression) : %d\n", t);

	// nonMaxi - 
	corner = nonMaximaSuppression(R, corner, height, width, 15);
	
	t = 0;
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			if (corner[y * width + x]) {
				t++;
			}
		}
	}
	//printf("Corner (After Suppression) : %d\n", t);

	free(lx);
	free(ly);
	free(R);

	
	// Visualization Corner
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			if (corner[y * width + x]) {
				pCenter.x = x;
				pCenter.y = y;
				c.val[0] = 255;
				c.val[1] = 0;
				c.val[2] = 0;
				circle(imgCorner, pCenter, radius, c, 5, 8, 0);
			}
		}
	}

	free(corner);
	return imgCorner;

}

// Harris Corner + Matching////////////////////////////////////////////////////////////////
vector<Point> GetCornerLocation(Mat img, int th, int wsize) {
	int x, y, xx, yy, height, width, sumx, sumy, IxIx, IxIy, IyIx, IyIy, det, tr;
	float max = -1, min = FLT_MAX;
	float k = 0.04;

	height = img.rows;
	width = img.cols;

	Mat imgCorner(height, width, CV_8UC3);

	int* lx = (int*)calloc(height * width, sizeof(int));
	int* ly = (int*)calloc(height * width, sizeof(int));
	int* corner = (int*)calloc(height * width, sizeof(int));

	float* R = (float*)calloc(height * width, sizeof(float));

	Scalar c;
	Point pCenter;
	int radius = 1;

	// Compute Gradient And Convert Channel
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			sumx = 0;
			sumy = 0;
			for (yy = y - MSIZE / 2; yy <= y + MSIZE / 2; yy++) {
				for (xx = x - MSIZE / 2; xx <= x + MSIZE / 2; xx++) {
					if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
						sumx += img.at<uchar>(yy, xx) * mask_x[(yy - (y - MSIZE / 2)) * MSIZE + (xx - (x - MSIZE / 2))];
						sumy += img.at<uchar>(yy, xx) * mask_y[(yy - (y - MSIZE / 2)) * MSIZE + (xx - (x - MSIZE / 2))];
					}
				}
				lx[y * width + x] = sumx;
				ly[y * width + x] = sumy;

				imgCorner.at<Vec3b>(y, x)[0] = img.at<uchar>(y, x);
				imgCorner.at<Vec3b>(y, x)[1] = img.at<uchar>(y, x);
				imgCorner.at<Vec3b>(y, x)[2] = img.at<uchar>(y, x);

			}
		}
	}

	// Compute R And Detect Corner 
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			IxIx = 0.0;
			IxIy = 0.0;
			IyIx = 0.0;
			IyIy = 0.0;
			for (yy = y - (wsize / 2); yy <= y + (wsize / 2); yy++) {
				for (xx = x - (wsize / 2); xx <= x + (wsize / 2); xx++) {
					if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
						IxIx += lx[yy * width + xx] * lx[yy * width + xx];
						IxIy += lx[yy * width + xx] * ly[yy * width + xx];
						IyIx += ly[yy * width + xx] * lx[yy * width + xx];
						IyIy += ly[yy * width + xx] * ly[yy * width + xx];
					}
				}
			}
			det = IxIx * IyIy - IxIy * IyIx;
			tr = IxIx + IyIy;
			R[y * width + x] = det - k * tr * tr;

			if (max < R[y * width + x]) max = R[y * width + x];
			if (min > R[y * width + x]) min = R[y * width + x];
		}
	}

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			R[y * width + x] = 255 * (R[y * width + x] - min) / (max - min);
			if (R[y * width + x] > th)	corner[y * width + x] = 1;
		}
	}

	int t = 0;
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			if (corner[y * width + x]) {
				t++;
			}
		}
	}
	printf("Corner (Before Suppression) : %d\n", t);

	// nonMaxi - 
	corner = nonMaximaSuppression(R, corner, height, width, 3);
	vector<Point> conerPoint;
	t = 0;
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			if (corner[y * width + x]) {
				Point c(x, y);
				conerPoint.push_back(c);
			}
		}
	}
	printf("Corner (After Suppression) : %d\n", t);

	free(lx);
	free(ly);
	free(R);

	return conerPoint;
}

float** GetCornerHOG(Mat img, int* corner, int wsize) {
	int i, j, x, y, xx, yy, height, width, idx, k = -1, BIN = 9;
	float sumx, sumy, mag, deg, rad;

	height = img.rows;
	width = img.cols;

	int mask_x[MSIZE * MSIZE] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
	int mask_y[MSIZE * MSIZE] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

	int Histo_row = height * width;
	float** HOG = (float**)calloc(Histo_row, sizeof(float*));

	for (j = 0; j < height; j++) {
		for (i = 0; i < width; i++) {
			k++;
			*(HOG + k) = (float*)calloc(BIN, sizeof(float));
		
			if (corner[j * width + i]) {
				for (y = j - wsize / 2; y <= j - wsize / 2; y++) {
					for (x = i - wsize / 2; x <= i + wsize / 2; x++) {
						sumx = 0;
						sumy = 0;

						for (yy = y - MSIZE / 2; yy <= y - MSIZE / 2; yy++) {
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
			//else Histogram =0
		}
	}

	float sum;
	for (y = 0; y < Histo_row; y++) {
		sum = 0.0;
		for (x = 0; x < BIN; x++)
			sum += HOG[y][x] * HOG[y][x];
		for (x = 0; x < BIN; x++) {
			if (!sum) break;
			else HOG[y][x] /= sqrt(sum);
		}
	}

	return HOG;
}
Point* MatchingCorner(Mat img1, Mat img2, int* corner1, int* corner2, int wsize) {
	int i, j, x, y, xx, yy, height1, width1, height2, width2, k, corner1_size, t, BIN = 9;
	float score = 0.0, Th, min;
	Point mp;

	height1 = img1.rows;
	width1 = img1.cols;
	height2 = img2.rows;
	width2 = img2.cols;

	Th = 0.05;
	Point* MatchingPoint = (Point*)calloc(2 * height1 * width1, sizeof(Point));

	float** HOG1, ** HOG2;
	HOG1 = GetCornerHOG(img1, corner1, wsize);
	HOG2 = GetCornerHOG(img2, corner2, wsize);

	for (y = 0; y < height1; y++) {
		for (x = 0; x < width1; x++) {
			min = FLT_MAX;
			mp.x = 0;
			mp.y = 0;
			if (corner1[y * width1 + x]) {
				for (yy = 0; yy < height2; yy++) {
					for (xx = 0; xx < width2; xx++) {
						if (corner2[yy * width2 + xx]) {
							score = 0.0;
							for (k = 0; k < BIN; k++) score += abs(HOG1[y * width1 + x][k] - HOG2[yy * width2 + xx][k]);

							if (min > score) {
								min = score;
								mp.x = xx;
								mp.y = yy;
							}
						}
					}
				}
				MatchingPoint[y * width1 + x + 0].x = x;
				MatchingPoint[y * width1 + x + 0].y = y;
				MatchingPoint[y * width1 + x + 1] = mp;
			}
		}
	}

	t = 0;
	for (y = 0; y < height1; y++) {
		for (x = 0; x < width1; x++) {
			if (MatchingPoint[y * width1 + x + 0].x != 0 &&
				MatchingPoint[y * width1 + x + 0].y != 0 &&
				MatchingPoint[y * width1 + x + 1].x != 0 &&
				MatchingPoint[y * width1 + x + 1].y != 0) {
				cout << MatchingPoint[y * width1 + x + 0] << ", " << MatchingPoint[y * width1 + x + 1] << endl;
				t++;
			}
	
		}
	}
	printf("%d Corner Matching!!!\n", t);


	freePtr(HOG1, height1 * width1);
	freePtr(HOG2, height2 * width2);

	return MatchingPoint;

}
Mat imgMerge(Mat img1, Mat img2, Point* matchpoint) {
	int x, y, height, width, newWidth;
	Point c1, c2;
	int radius = 1, t = 0;
	Scalar Green, Blue;
	Green.val[0] = 0;
	Green.val[1] = 255;
	Green.val[2] = 0;

	Blue.val[0] = 255;
	Blue.val[1] = 0;
	Blue.val[2] = 0;

	// 비교하는 두 사진의 크기가 같다고 생각
	height = img1.rows;
	width = img1.cols;


	newWidth = img1.cols + img2.cols;

	Mat imgProduct = Mat::zeros(img1.rows, newWidth, CV_8UC3);

	for (y = 0; y < img1.rows; y++) {
		for (x = 0; x < img1.cols; x++) {
			imgProduct.at<Vec3b>(y, x)[0] = img1.at<uchar>(y, x);
			imgProduct.at<Vec3b>(y, x)[1] = img1.at<uchar>(y, x);
			imgProduct.at<Vec3b>(y, x)[2] = img1.at<uchar>(y, x);
		}
	}

	for (y = 0; y < img2.rows; y++) {
		for (x = img1.cols; x < newWidth; x++) {
			imgProduct.at<Vec3b>(y, x)[0] = img2.at<uchar>(y, x - img1.cols);
			imgProduct.at<Vec3b>(y, x)[1] = img2.at<uchar>(y, x - img1.cols);
			imgProduct.at<Vec3b>(y, x)[2] = img2.at<uchar>(y, x - img1.cols);
		}
	}

	float distanceC1C2;
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			c1 = matchpoint[y * width + x + 0];
			c2 = matchpoint[y * width + x + 1];

			distanceC1C2 = sqrt(pow(c1.x - c2.x, 2) + pow(c1.y - c2.y, 2));

			if (distanceC1C2 < 50) {
				if (c1.x != 0 && c1.y != 0 && c2.x != 0 && c2.y != 0) {
					c2.x = matchpoint[y * width + x + 1].x + width;
					c2.y = matchpoint[y * width + x + 1].y;

					circle(imgProduct, c1, radius, Green, 2, 8, 0);
					circle(imgProduct, c2, radius, Green, 2, 8, 0);
					line(imgProduct, c1, c2, Blue, 2, 9, 0);
					t++;
				}
			}
		}
	}

	printf("%d Line!!!\n", t);


	return imgProduct;
}
