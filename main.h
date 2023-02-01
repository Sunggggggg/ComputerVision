#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <math.h>
#include <opencv2/photo.hpp>
#include "utils.h"


#define MSIZE 3
#define PI 3.141592
using namespace cv;
using namespace std;



Mat MakeGrayPart(Mat img);
Mat ImgResize(Mat imgGray, int flag);
Mat ImgRotation(Mat imgGray, int flag);
Mat ImgResize_Forward(Mat imgGray);

Mat EdgeMagnitude(Mat imgGray);
Mat EdgeMagnitude_Extra(Mat imgGray);

// HOG
float** GetHistogram(Mat img, int BLK, int BIN);
float* GetLBPHisto(Mat lbp, int bin);
void freePtr(float**a, int row);

void printHOG(float** a, int row, int col);
void Savecsv(float** a, int height, int width);
float CompareHistogram(float** a, float** b, int height, int width);

// Corner
Mat GetGaussianFilter(Mat img);
Mat GetHarrisCorner(Mat imgGray, int th, int wsize);
vector<Point> GetCornerLocation(Mat img, int th, int wsize);
Mat imgMerge(Mat img1, Mat img2, Point* a);
Point* MatchingCorner(Mat img1, Mat img2, int* corner1, int* corner2, int wsize);


int VideoSave();
int CornerVideos();
int GetORB(string filename);
void GetOpticalFlow();

void GetFaceDetect(string filename);

// Processing
Mat CropImg(Mat img, Point tl, Point rb);

// Sim
Mat GeneralGetFaceDetect(Mat imgRef, Mat imgTar);
float GetSimilarity(float** ref, float** tar, int Hrow, int bin, int flag);

Mat GetLBP(Mat img, int blk);
float** GetHistogram_C(Mat lbp, int size, int bin);

Mat SLICsegmentation(Mat image, int m_spcount, double m_compactness);
Mat GetIpainting(Mat imgDoodle);

vector<Mat> GetSVD(Mat matInput);
Mat GetH(vector<Point> xy, vector<Point> pxy);
Mat GetHomography(Mat img, Mat H, int height, int width);

void GraphCut_mask(Mat input, Mat input_with_mask);

vector<Point> GetGraphCut(Mat img, Mat imgMask, Vec3b myColor, float rate);
vector<Point> ROIPoint(Mat img, vector<Point> pt);