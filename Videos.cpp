#define _CRT_SECURE_NO_WARNINGS
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<Windows.h>


using namespace cv;

Mat GetHarrisCorner(Mat img, int th, int wsize);

int VideoSave() {
	VideoCapture capture(0);
	Mat frame, frameGray;
	int frameN = 0;
	char filename[100];

	if (!capture.isOpened()) {
		printf("Open Fail...!!!!\n");
		return -1;
	}

	while (1) {
		capture >> frame;

		imshow("Video", frame);
		if (waitKey(30) >= 0)break;

		sprintf(filename, "./image/image%04d.bmp", frameN++);
		imwrite(filename, frame);
	}

}

int CornerVideos() {
	VideoCapture capture(0);
	Mat frame, frameGray;
	int frameN = 0;
	char filename[100];

	LARGE_INTEGER freq, start, stop;
	double diff, avg = 0;


	if (!capture.isOpened()) {
		printf("Open Fail...!!!!\n");
		return -1;
	}

	while (1) {
		capture >> frame;
		cvtColor(frame, frameGray, COLOR_BGR2GRAY);

		Mat Cornerframe(frameGray.rows, frameGray.cols, CV_8UC1);

		QueryPerformanceFrequency(&freq);
		QueryPerformanceCounter(&start);

		Cornerframe = GetHarrisCorner(frameGray, 50000, 3);

		QueryPerformanceCounter(&stop);
		diff = (double)(stop.QuadPart - start.QuadPart) / freq.QuadPart;
		avg += diff;
		
		printf("Operating time (Average) : %f\n",frameN / diff);

		imshow("Video", Cornerframe);
		if (waitKey(30) >= 0)break;

		frameN++;
	}
}
