#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <stdio.h>
#include "utils.h"

#define MSIZE 3
#define PI 3.141592
using namespace cv;
using namespace std;

// ORB settings
int ORB_MAX_KPTS = 1500;
float ORB_SCALE_FACTOR = 1.2;
int ORB_PYRAMID_LEVELS = 4;
float ORB_EDGE_THRESHOLD = 31.0;
int ORB_FIRST_PYRAMID_LEVEL = 0;
int ORB_WTA_K = 2;
int ORB_PATCH_SIZE = 31;

// Some image matching options
float MIN_H_ERROR = 2.50f; // Maximum error in pixels to accept an inlier
float DRATIO = 0.80f;

int GetORB(string filename) {
	Mat img1, img1_32, img2, img2_32;
	string img_path1, img_path2, homography_path;
	double t1 = 0.0, t2 = 0.0;
	vector<KeyPoint> kpts1_orb;
	Mat desc1_orb;

	int nmatches_orb = 0, ninliers_orb = 0, noutliers_orb = 0, nkpts1_orb = 0, nkpts2_orb = 0;
	float ratio_orb = 0.0;

	// Set video
	VideoCapture capture(0);
	if (!capture.isOpened()) {
		printf("Open Fail...!!!!\n");
		return -1;
	}

	// Define DescriptorMatcher
	Ptr<DescriptorMatcher> matcher_l2 = DescriptorMatcher::create("BruteForce");
	Ptr<DescriptorMatcher> matcher_l1 = DescriptorMatcher::create("BruteForce-Hamming");
	// Define keypoint detect
	Ptr<ORB> orb = ORB::create(ORB_MAX_KPTS, ORB_SCALE_FACTOR, ORB_PYRAMID_LEVELS,
		ORB_EDGE_THRESHOLD, ORB_FIRST_PYRAMID_LEVEL, ORB_WTA_K, ORB::HARRIS_SCORE,
		ORB_PATCH_SIZE);

	double torb = 0.0; // Create the L2 and L1 matchers

	img1 = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);		// GrayScale

	// Resizeing + 32F(Float) 형태로, 3개의 채널
	Size reSize(300, 300);
	resize(img1, img1, reSize, 0, 0, INTER_LINEAR);
	img1.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);			// 0 ~ 1 사이 값으로 나타냄.
	
	orb->detectAndCompute(img1, noArray(), kpts1_orb, desc1_orb, false);	// 

	Mat img1_rgb_orb = Mat(Size(img1.cols, img1.rows), CV_8UC3);
	cvtColor(img1, img1_rgb_orb, COLOR_GRAY2BGR);
	draw_keypoints(img1_rgb_orb, kpts1_orb);

	
	while (1) {
		Mat desc2_orb;
		vector<KeyPoint> kpts2_orb;
		
		capture >> img2;
		// Resizeing + 32F(Float) 형태로, 3개의 채널
		resize(img2, img2, reSize, 0, 0, INTER_LINEAR);
		cvtColor(img2, img2, COLOR_BGR2GRAY);
		img2.convertTo(img2_32, CV_32F, 1.0 / 255.0, 0);

		orb->detectAndCompute(img2, noArray(), kpts2_orb, desc2_orb, false);

		// Matching 
		vector<vector<DMatch>> dmatches_orb;
		matcher_l1->knnMatch(desc1_orb, desc2_orb, dmatches_orb, 2);

		vector<Point2f> matches_orb, inliers_orb;
		matches2points_nndr(kpts1_orb, kpts2_orb, dmatches_orb, matches_orb, DRATIO);
		compute_inliers_ransac(matches_orb, inliers_orb, MIN_H_ERROR, false);

		// Number of Keypoints Image
		nkpts1_orb = kpts1_orb.size();
		nkpts2_orb = kpts2_orb.size();

		// Number of Matches
		nmatches_orb = matches_orb.size() / 2;
		ninliers_orb = inliers_orb.size() / 2;
		noutliers_orb = nmatches_orb - ninliers_orb;
		ratio_orb = 100.0 * (float)(ninliers_orb) / (float)(nmatches_orb);		// 정확도의 지표가 될 수도 있음.

		Mat img2_rgb_orb = Mat(Size(img2.cols, img1.rows), CV_8UC3);
		Mat img_com_orb = Mat(Size(img1.cols * 2, img1.rows), CV_8UC3);

		cvtColor(img2, img2_rgb_orb, COLOR_GRAY2BGR);

		// Video Point
		draw_keypoints(img2_rgb_orb, kpts2_orb);
		draw_inliers(img1_rgb_orb, img2_rgb_orb, img_com_orb, inliers_orb, 0);

		imshow("ORB", img_com_orb);
		if (waitKey(30) >= 0)break;

		cout << "ORB Results" << endl;
		cout << "**************************************" << endl;
		cout << "Number of Keypoints Image 1: " << nkpts1_orb << endl;
		cout << "Number of Keypoints Image 2: " << nkpts2_orb << endl;
		cout << "Number of Matches: " << nmatches_orb << endl;
		cout << "Number of Inliers: " << ninliers_orb << endl;
		cout << "Number of Outliers: " << noutliers_orb << endl;
		cout << "Inliers Ratio: " << ratio_orb << endl;
		cout << endl;
	}
}