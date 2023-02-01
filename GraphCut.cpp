#include"GraphCut.h"

void GraphCut_mask(Mat input, Mat input_with_mask) {
    int x, y;
    int height = input.rows;
    int width = input.cols;


    Rect rectangle(0, 0, width - 1, height - 1);
    Mat mask = Mat::ones(height, width, CV_8UC1) * GC_PR_BGD;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            if (input_with_mask.at<Vec3b>(y, x)[0] == 255 && input_with_mask.at<Vec3b>(y, x)[1] == 255 && input_with_mask.at<Vec3b>(y, x)[2] == 0) {
                mask.at<uchar>(y, x) = GC_BGD;
            }
            else if (input_with_mask.at<Vec3b>(y, x)[0] == 0 && input_with_mask.at<Vec3b>(y, x)[1] == 0 && input_with_mask.at<Vec3b>(y, x)[2] == 255) {
                mask.at<uchar>(y, x) = GC_FGD;
            }
        }
    }

    Mat final_result = Mat::zeros(height, width, CV_8UC3);
    Mat mask_result = Mat::zeros(height, width, CV_8UC1);

    Mat bg, fg;

    final_result = input.clone();

    cv::grabCut(input, mask, rectangle, bg, fg, 8, GC_INIT_WITH_MASK);

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            if (mask.at<uchar>(y, x) == GC_FGD || mask.at<uchar>(y, x) == GC_PR_FGD) {
                mask_result.at<uchar>(y, x) = 255;
            }
            else {
                final_result.at<Vec3b>(y, x)[0] = 255;
                final_result.at<Vec3b>(y, x)[1] = 255;
                final_result.at<Vec3b>(y, x)[2] = 255;
            }
        }
    }


#ifdef MASK2
    Mat mask2 = Mat::ones(height, width, CV_8UC1) * GC_PR_BGD;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            if (input_with_mask.at<Vec3b>(y, x)[0] == 255 && input_with_mask.at<Vec3b>(y, x)[1] == 0 && input_with_mask.at<Vec3b>(y, x)[2] == 0) {
                mask2.at<uchar>(y, x) = GC_PR_BGD;
            }
            else if (input_with_mask.at<Vec3b>(y, x)[0] == 0 && input_with_mask.at<Vec3b>(y, x)[1] == 0 && input_with_mask.at<Vec3b>(y, x)[2] == 255) {
                mask2.at<uchar>(y, x) = GC_PR_FGD;
            }
        }
    }

    Mat final_result2 = Mat::zeros(height, width, CV_8UC3);
    Mat mask_result2 = Mat::zeros(height, width, CV_8UC1);

    final_result2 = input.clone();
    grabCut(input, mask2, rectangle, bg, fg, 8, GC_INIT_WITH_MASK);
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            if (mask2.at<uchar>(y, x) == GC_FGD || mask2.at<uchar>(y, x) == GC_PR_FGD) {
                mask_result2.at<uchar>(y, x) = 255;
            }
            else {
                final_result2.at<Vec3b>(y, x)[0] = 255;
                final_result2.at<Vec3b>(y, x)[1] = 255;
                final_result2.at<Vec3b>(y, x)[2] = 255;
            }
        }
    }
#endif

    imshow("res", final_result);
    imshow("mask_result", mask_result);
    waitKey(0);

}