#include "FinalProject.h"

// FOR SVD
Mat Reduce_fraction(Mat src) {
	int x, y, row, col;
	row = src.rows;
	col = src.cols;

	Mat result(row, col, CV_32FC1);
	float* localmin = (float*)calloc(row, sizeof(float));
	float min;

	for (y = 0; y < row; y++) {
		min = FLT_MAX;
		for (x = 0; x < col; x++) {
			if (src.at<float>(y, x) < min) min = src.at<float>(y, x);
		}
		if (min < 0) min = abs(min);
		localmin[y] = min;
	}

	for (y = 0; y < row; y++) {
		for (x = 0; x < col; x++) {
			result.at<float>(y, x) = src.at<float>(y, x) / localmin[y];
		}
	}
	return result;
}
Mat trans(Mat src) {
	int x, y, height, width;
	height = src.rows;
	width = src.cols;

	Mat result(width, height, CV_32FC1);

	for (x = 0; x < width; x++) {
		for (y = 0; y < height; y++) {
			result.at<float>(x, y) = src.at<float>(y, x);
		}
	}

	return result;

}
float GetVecMag(Mat src) {
	int x, y, height, width;
	height = src.rows;
	width = src.cols;		// 1
	float sum = 0.0;


	for (x = 0; x < width; x++) {
		for (y = 0; y < height; y++) {
			sum += (src.at<float>(y, x) * src.at<float>(y, x));
		}
	}

	return sqrt(sum);
}
Mat MatProduct(Mat a, Mat b) {
	int x, y, n, i, j, nrow, ncol;
	float temp;
	nrow = a.rows;
	ncol = b.cols;
	n = a.cols;		// = b.rows;

	Mat result(nrow, ncol, CV_32FC1);
	for (y = 0; y < nrow; y++) {
		for (x = 0; x < ncol; x++) {
			temp = 0;
			for (j = 0; j < n; j++) {
				temp += a.at<float>(y, j) * b.at<float>(j, x);
			}
			result.at<float>(y, x) = temp;
		}
	}
	return result;
}
Mat pushCol(Mat a, Mat b, int i) {
	int x;
	int n = b.rows;

	for (x = 0; x < n; x++) {
		a.at<float>(x, i) = b.at<float>(x, 0);
	}

	return a;
}
Mat GramShumidt(Mat eVector) {
	int i, j;
	int M = eVector.rows;

	Mat u(M, M, CV_32FC1);
	Mat vT, MP, temp;
	Mat vector_u = Mat::zeros(M, 1, CV_32FC1);
	float mag;
	u = eVector.clone();

	// ±×¶÷ ½´¹ÌÆ® eigVectors -> U
	for (i = 0; i < M; i++) {
		for (j = i - 1; j >= 0; j--) {
			MP = MatProduct(eVector.row(i), u.col(j));
			vector_u -= MP.at<float>(0, 0) * u.col(j);
		}
		temp = trans(eVector.row(i)) - vector_u;

		mag = GetVecMag(temp);
		pushCol(u, temp / mag, i);
	}

	return u;
}
vector<Mat> GetSVD(Mat matInput) {
	int y;
	vector<Mat> result(3);
	Mat AT = trans(matInput);

	Mat eigValues, eigVectors;
	cv::eigen(matInput * AT, eigValues, eigVectors);
	eigVectors = Reduce_fraction(eigVectors);

	Mat U;
	U = GramShumidt(eigVectors);

	Mat eigValues_V, eigVectors_V;
	cv::eigen(AT * matInput, eigValues_V, eigVectors_V);
	eigVectors_V = Reduce_fraction(eigVectors_V);

	Mat V;
	V = GramShumidt(eigVectors_V);

	// Find S
	Mat VT = trans(V);
	int srow = U.cols;
	int scol = VT.rows;

	Mat S = Mat::zeros(srow, scol, CV_32FC1);
	for (y = 0; y < srow; y++) {
		S.at<float>(y, y) = sqrt(eigValues.at<float>(y, 0));
	}

	result[0] = U;
	result[1] = S;
	result[2] = V;

	return result;
}

// FOR Homography
Mat GetH(vector<Point> xy, vector<Point> pxy) {
	float A[8][9] = {
		{-xy[0].x,	-xy[0].y,	-1,		0,			0,			 0,	xy[0].x * pxy[0].x,	 xy[0].y * pxy[0].x, pxy[0].x },
		{0,			0,			0,		-xy[0].x,	 -xy[0].y,  -1,	xy[0].x * pxy[0].y,	 xy[0].y * pxy[0].y, pxy[0].y },
		{-xy[1].x,	-xy[1].y,	-1,		0,			0,			 0,	xy[1].x * pxy[1].x,	 xy[1].y * pxy[1].x, pxy[1].x },
		{0,			0,			0,		-xy[1].x,	 -xy[1].y,  -1,	xy[1].x * pxy[1].y,	 xy[1].y * pxy[1].y, pxy[1].y },
		{-xy[2].x,	-xy[2].y,	-1,		0,			0,			 0,	xy[2].x * pxy[2].x,	 xy[2].y * pxy[2].x, pxy[2].x },
		{0,			0,			0,		-xy[2].x,	 -xy[2].y,  -1,	xy[2].x * pxy[2].y,	 xy[2].y * pxy[2].y, pxy[2].y },
		{-xy[3].x,	-xy[3].y,	-1,		0,			0,			 0,	xy[3].x * pxy[3].x,	 xy[3].y * pxy[3].x, pxy[3].x },
		{0,			0,			0,		-xy[3].x,	 -xy[3].y,  -1,	xy[3].x * pxy[3].y,	 xy[3].y * pxy[3].y, pxy[3].y },
	};
	Mat input(8, 9, CV_32FC1, &A);
	vector<Mat> USV(3);
	USV = GetSVD(input);

	Mat V = USV[2].clone();
	Mat temp = V.col(8);
	int n = sqrt(V.col(8).rows);
	Mat H(n, n, CV_32FC1);
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < n; i++) {
			H.at<float>(i, j) = temp.at<float>(j * n + i, 0);
		}
	}

	return H;
}
Mat GetHomography(Mat img, Mat H, int height, int width) {
	uchar p1, p2, p3, p4;
	int x, y, rx, ry;
	float px, py, dx, dy;

	Mat result(height, width, CV_8UC3);

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			Mat pState(1, 3, CV_32FC1);
			Mat State(1, 3, CV_32FC1);
			pState.at<float>(0, 0) = x;
			pState.at<float>(0, 1) = y;
			pState.at<float>(0, 2) = 1;

			State = pState * H.inv();		// Backward
			
			px = State.at<float>(0, 0) / State.at<float>(0, 2);
			py = State.at<float>(0, 1) / State.at<float>(0, 2);
			
			if (px >= 0 && px < img.cols && py >= 0 && py < img.rows) {
				rx = int(px);
				ry = int(py);

				dx = px - rx;
				dy = py - ry;

				p1 = img.at<Vec3b>(ry, rx)[0];
				p2 = img.at<Vec3b>(ry, rx + 1)[0];
				p3 = img.at<Vec3b>(ry + 1, rx)[0];
				p4 = img.at<Vec3b>(ry + 1, rx + 1)[0];

				result.at<Vec3b>(y, x)[0] = p1 * (1 - dx) * (1 - dy) + p2 * dx * (1 - dy) + p3 * (1 - dx) * dy + p4 * dx * dy;

				p1 = img.at<Vec3b>(ry, rx)[1];
				p2 = img.at<Vec3b>(ry, rx + 1)[1];
				p3 = img.at<Vec3b>(ry + 1, rx)[1];
				p4 = img.at<Vec3b>(ry + 1, rx + 1)[1];

				result.at<Vec3b>(y, x)[1] = p1 * (1 - dx) * (1 - dy) + p2 * dx * (1 - dy) + p3 * (1 - dx) * dy + p4 * dx * dy;

				p1 = img.at<Vec3b>(ry, rx)[2];
				p2 = img.at<Vec3b>(ry, rx + 1)[2];
				p3 = img.at<Vec3b>(ry + 1, rx)[2];
				p4 = img.at<Vec3b>(ry + 1, rx + 1)[2];

				result.at<Vec3b>(y, x)[2] = p1 * (1 - dx) * (1 - dy) + p2 * dx * (1 - dy) + p3 * (1 - dx) * dy + p4 * dx * dy;
			}
			else {
				result.at<Vec3b>(y, x)[0] = 0;
				result.at<Vec3b>(y, x)[1] = 0;
				result.at<Vec3b>(y, x)[2] = 0;
			}
			
		}
	}
	return result;
}

// FOR Grapcut
vector<Point> GetGraphCut(Mat img, Mat imgMask, Vec3b myColor, float rate) {
	int x, y;
	int height = img.rows;
	int width = img.cols;

	Rect rectangle(0, 0, width - 1, height - 1);
	Mat mask = Mat::ones(height, width, CV_8UC1) * GC_PR_BGD;

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			if (imgMask.at<Vec3b>(y, x) == myColor) {
				mask.at<uchar>(y, x) = GC_FGD;
			}
		}
	}

	Mat mask_result = Mat::zeros(height, width, CV_8UC1);
	Mat bg, fg;

	Point Center(img.cols / 2, img.rows / 2);
	float dist;
	int dx, dy;
	vector<Point> fgdPoint;

	cv::grabCut(img, mask, rectangle, bg, fg, 5, GC_INIT_WITH_MASK);
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			if (mask.at<uchar>(y, x) == GC_FGD || mask.at<uchar>(y, x) == GC_PR_FGD) {
				dx = x - Center.x;
				dy = y - Center.y;
				dist = sqrt(dx * dx + dy * dy);
				if (dist < img.cols / rate) {
					mask_result.at<uchar>(y, x) = 255;
					Point C(x, y);
					fgdPoint.push_back(C);
				}
			}
		}
	}

	imshow("mask_result", mask_result);
	return fgdPoint;
}
vector<Point> ROIPoint(Mat img, vector<Point> pt) {
	int k;

	vector<Point> ReorderPt(4);
	vector<int> sumPoints, subPoints;
	for (k = 0; k < pt.size(); k++) {
		sumPoints.push_back(pt[k].x + pt[k].y);
		subPoints.push_back(pt[k].x - pt[k].y);
	}

	int minsub = INT_MAX, maxsub = -1, minsum = INT_MAX, maxsum = -1;
	for (k = 0; k < pt.size(); k++) {
		if (sumPoints[k] < minsum) minsum = sumPoints[k];
		if (sumPoints[k] > maxsum) maxsum = sumPoints[k];
		if (subPoints[k] < minsub) minsub = subPoints[k];
		if (subPoints[k] > maxsub) maxsub = subPoints[k];
	}
	
	for (k = 0; k < pt.size(); k++) {
		if (sumPoints[k] == minsum)	ReorderPt[0] = pt[k];
		if (sumPoints[k] == maxsum)	ReorderPt[2] = pt[k];
		if (subPoints[k] == minsub)	ReorderPt[3] = pt[k];
		if (subPoints[k] == maxsub)	ReorderPt[1] = pt[k];
	}


	return ReorderPt;
}
