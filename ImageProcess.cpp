#include "ImageProcess.h"

Mat CropImg(Mat img, Point tl, Point rb) {
	int x, y;
	int startx, endx;
	int starty, endy;

	startx = tl.x;
	endx = rb.x;

	starty = tl.y;
	endy = rb.y;

	Mat result(endy - starty, endx - startx, CV_8UC1);

	for (y = starty; y < endy; y++) {
		for (x = startx; x < endx; x++) {
			result.at<uchar>(y - starty, x - startx) = img.at<uchar>(y, x);
		}
	}
	return result;
}