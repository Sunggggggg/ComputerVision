#include "FaceVerifcation.h"
#include "HOGUnifrom.h"
#include "flag.h"
#include "ldmarkmodel.h"

Mat CropImg(Mat img, Point tl, Point rb);
float GetSimilarity(float** ref, float** tar, int Hrow, int bin, int flag);
void freePtr(float** a, int row);

Mat GetLBP(Mat img, int blk) {
	int i, x1, x2, y1, y2, height, width;
	uchar center, dec;
	height = img.rows;
	width = img.cols;
	Mat LBP = Mat::zeros(height, width, CV_8UC1);
	int PattenSize = (blk - 1) * 4;


	for (y1 = 0; y1 < height; y1++) {
		for (x1 = 0; x1 < width; x1++) {
			//printf("%d %d\n", x1, y1);
			int* temp = (int*)calloc(PattenSize, sizeof(int));
			center = img.at<uchar>(y1, x1);
			
			for (y2 = y1 - blk /2; y2 <= y1 + blk / 2; y2++) {
				for (x2 = x1 - blk / 2; x2 <= x1 + blk / 2; x2++) {
					if (y2 >= 0 && y2 < height && x2 >= 0 && x2 < width) {
						if ((y2 - y1) == -(blk / 2) || (y2 - y1) == (blk / 2) || (x2 - x1) == -(blk / 2) || (x2 - x1) == (blk / 2)) {
							if (center > img.at<uchar>(y2, x2)) temp[(y2 - (y1 - (blk / 2))) * blk + (x2 - (x1 - (blk / 2)))] = 1;
						}
					}
				}
			}
			dec = 0;
			// 10Áø¼öÈ­
			for (i = 0; i < PattenSize; i++) dec += uchar(pow(2, i) * temp[i]);
			LBP.at<uchar>(y1, x1) = dec;

			free(temp);
		}
	}

	return LBP;
}
float** GetHistogram_C(Mat lbp, int rate, int bin) {
	int x1, x2, y1, y2, height, width, widthBLK, heightBLK;

	height = lbp.rows;
	width = lbp.cols;

	widthBLK = width / rate;
	heightBLK = height / rate;

	int Hrow = ((height - heightBLK) / (heightBLK / 2) + 1)
		* ((width - widthBLK) / (widthBLK / 2) + 1);

	int k = -1;
	float** HOGRef = (float**)calloc(Hrow, sizeof(float*));

	// Get HOG
	for (y1 = 0; y1 <= height - heightBLK; y1 += heightBLK / 2) {
		for (x1 = 0; x1 <= width - widthBLK; x1 += widthBLK / 2) {
			k++;
			HOGRef[k] = (float*)calloc(bin, sizeof(float));

			for (y2 = y1; y2 < y1 + heightBLK; y2++) {
				for (x2 = x1; x2 < x1 + widthBLK; x2++) {
					// HOGRef[k][lbp.at<uchar>(y2, x2)] += 1;
					HOGRef[k][uniform[lbp.at<uchar>(y2, x2)]] += 1;
				}
					
			}
		}
	}

	// Lorm-
	float sum;
	for (y1 = 0; y1 < Hrow; y1++) {
		sum = 0.0;
		for (x1 = 0; x1 < bin; x1++)
			sum += HOGRef[y1][x1] * HOGRef[y1][x1];

		if (!sum) {
			for (x1 = 0; x1 < bin; x1++)
				HOGRef[y1][x1] /= sqrt(sum);
		}
		else
			HOGRef[y1][x1] = 0;
	}

	return HOGRef;
}

float* GetLBPHisto(Mat lbp, int bin) {
	int x1, y1, height, width;
	float* HOG = (float*)calloc(bin, sizeof(float));

	height = lbp.rows;
	width = lbp.cols;

	// Get HOG
	for (y1 = 0; y1 < height; y1++) {
		for (x1 = 0; x1 < width; x1++) {
			HOG[uniform[lbp.at<uchar>(y1, x1)]] += 1;
		}
	}

	// Lorm-
	float sum = 0.0;

	for (x1 = 0; x1 < bin; x1++)
		sum += HOG[x1] * HOG[x1];

	if (!sum) {
		for (x1 = 0; x1 < bin; x1++)
			HOG[x1] /= sqrt(sum);
	}
	else
		HOG[x1] = 0;

	return HOG;
}

int FaceVerification_basic() {
	int bin, blkRate, Hrow, heightRef, widthRef, heightBLK, widthBLK;
	float** HOGRef = NULL;
	Mat frame, imgRef, imgRefGray, LBPRef, frameGray, LBHTar, imgExtract;
	int i, flag = 1;
	float Th = 0.97;
	float corelation;

	CascadeClassifier cascade;
	cascade.load("C:/Users/sky/OpenCV_Version/opencv_345/sources/data/lbpcascades/lbpcascade_frontalface.xml");

	VideoCapture capture(0);
	if (!capture.isOpened()) {
		printf("Open Fail...!!!!\n");
		return -1;
	}

	bin = 59;
	blkRate = 3;

	// 1. Reference face
	while (flag) {
		capture >> imgRef;
		vector<Rect> facesRef;
		cascade.detectMultiScale(imgRef, facesRef, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));

		Point rb, tl;
		for (i = 0; i < facesRef.size(); i++) {
			rb.x = facesRef[i].x + facesRef[i].width;
			rb.y = facesRef[i].y + facesRef[i].height;

			tl.x = facesRef[i].x;
			tl.y = facesRef[i].y;

			rectangle(imgRef, tl, rb, Scalar(0, 255, 0), 3, 8, 0);
		}

		imshow("Video", imgRef);

		// Capture When Face detect size = 1
		if ((waitKey(30) == 'c') && (facesRef.size() == 1)) {
			heightRef = rb.y - tl.y;
			widthRef = rb.x - tl.x;

			widthBLK = widthRef / blkRate;
			heightBLK = heightRef / blkRate;
			Hrow = ((heightRef - heightBLK) / (heightBLK / 2) + 1)
				* ((widthRef - widthBLK) / (widthBLK / 2) + 1);

			printf("-------------------------------------------------------------\n");
			printf("Reference image width : %d\t height : %d\n", widthBLK, heightBLK);
			printf("Histogram Row : %d\n", Hrow);
			printf("-------------------------------------------------------------\n");

			cvtColor(imgRef, imgRefGray, COLOR_BGR2GRAY);
			imgExtract = CropImg(imgRefGray, tl, rb);
			resize(imgExtract, imgExtract, Size(widthBLK, heightBLK));

			LBPRef = GetLBP(imgExtract, 3);
			HOGRef = GetHistogram_C(LBPRef, blkRate, bin);
			flag = 0;

			printf("Reference Capture !!\n");
			destroyWindow("Video");
		}
	}

	imshow("imgRef", imgExtract);
	waitKey(0);

	while (1) {
		capture >> frame;
		vector<Rect> faces;
		cascade.detectMultiScale(frame, faces, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));

		// 2. Detect Face
		vector<float**> HOGTar(faces.size());
		for (i = 0; i < faces.size(); i++) {
			Point rb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
			Point tl(faces[i].x, faces[i].y);

			// 3. Crop image
			cvtColor(frame, frameGray, COLOR_BGR2GRAY);
			imgExtract = CropImg(frameGray, tl, rb);
			resize(imgExtract, imgExtract, Size(widthBLK, heightBLK));

			LBHTar = GetLBP(imgExtract, 3);
			HOGTar[i] = GetHistogram_C(LBHTar, blkRate, bin);

			// 4. GetSimilarity
			corelation = GetSimilarity(HOGRef, HOGTar[i], Hrow, bin, COSINE);

			if (corelation > Th) rectangle(frame, tl, rb, Scalar(0, 255, 0), 3, 8, 0);
			else rectangle(frame, tl, rb, Scalar(0, 0, 255), 3, 8, 0);

			imshow("imgExtract", imgExtract);
			freePtr(HOGTar[i], Hrow);
		}

		imshow("frame", frame);
		if (waitKey(30) >= 0)break;
	}

	freePtr(HOGRef, Hrow);
	return 0;
}

int FaceVerification_SDM() {
	int x1 = 0, y1 = 0, x2, y2, i, j, widhtblk, heightblk, bin, numLandmarks, ENM;
	Mat img, imgGray, current_shape;
	float corelation;

	VideoCapture caputure(0);
	if (!caputure.isOpened()) {
		cout << "Camera opening failed..." << endl;
		return 0;
	}

	ldmarkmodel model;
	string modelFilePath = "roboman-landmark-model.bin";
	while (!load_ldmarkmodel(modelFilePath, model)) {
		cout << "Error!!\n" << endl;
	}

	// Reference
	Mat imgRef = imread("./images/faceRef.PNG");
	cvtColor(imgRef, imgGray, CV_BGR2GRAY);

	Vec3d headpose;
	model.track(imgRef, current_shape);
	model.EstimateHeadPose(current_shape, headpose);
	model.drawPose(imgRef, current_shape, 50);

	numLandmarks = current_shape.cols / 2;
	ENM = 67 - 27 + 1;
	std::printf("ENM : %d\n", ENM);

	x1 = current_shape.at<float>(36);
	y1 = current_shape.at<float>(36 + numLandmarks);

	Point LE(x1, y1);

	x1 = current_shape.at<float>(45);
	y1 = current_shape.at<float>(45 + numLandmarks);

	Point RE(x1, y1);

	int distance = sqrt(pow((RE.x - LE.x), 2) + pow((RE.y - LE.y), 2));
	int rate = 7;

	heightblk = distance / rate;
	widhtblk = distance / rate;

	std::printf("distance : %d\n", distance);
	std::printf("heightblk : %d, widhtblk : %d\n", heightblk, widhtblk);

	//heightblk = widhtblk = 16;
	bin = 59;
	cvtColor(imgRef, imgGray, CV_BGR2GRAY);

	float** HOGRef = (float**)calloc(ENM, sizeof(float*));
	vector<Mat> LandmarkLBP(ENM);
	Mat imgTemp(heightblk, widhtblk, CV_8UC1);

	for (j = 27; j < numLandmarks; j++) {
		x1 = current_shape.at<float>(j);
		y1 = current_shape.at<float>(j + numLandmarks);

		circle(imgRef, Point(x1, y1), 2, Scalar(0, 0, 255), -1);

		HOGRef[j - 27] = (float*)calloc(bin, sizeof(float));

		for (y2 = y1 - heightblk / 2; y2 < y1 + heightblk / 2; y2++) {
			for (x2 = x1 - widhtblk / 2; x2 < x1 + widhtblk / 2; x2++) {
				imgTemp.at<uchar>((y2 - y1) + heightblk / 2, (x2 - x1) + widhtblk / 2) = imgGray.at<uchar>(y2, x2);
				LandmarkLBP[j - 27] = GetLBP(imgTemp, 3);
			}
		}
		HOGRef[j - 27] = GetLBPHisto(LandmarkLBP[j - 27], bin);
	}

	imshow("imgRef", imgRef);
	waitKey(0);

	while (1) {
		caputure >> img;

		Vec3d eavTar;
		model.track(img, current_shape);
		model.EstimateHeadPose(current_shape, eavTar);
		model.drawPose(img, current_shape, 50);

		numLandmarks = current_shape.cols / 2;

		x2 = current_shape.at<float>(36);
		y2 = current_shape.at<float>(36 + numLandmarks);

		Point LE(x2, y2);

		x2 = current_shape.at<float>(45);
		y2 = current_shape.at<float>(45 + numLandmarks);

		Point RE(x2, y2);

		int distance = sqrt(pow((RE.x - LE.x), 2) + pow((RE.y - LE.y), 2));

		heightblk = distance / rate;
		widhtblk = distance / rate;

		std::printf("distance : %d\n", distance);
		std::printf("heightblk : %d, widhtblk : %d\n", heightblk, widhtblk);

		cvtColor(img, imgGray, CV_BGR2GRAY);
		Mat imgTemp(heightblk, widhtblk, CV_8UC1);

		float** LBPTar = (float**)calloc(numLandmarks, sizeof(float*));
		vector<Mat> LandmarkLBPTar(numLandmarks);

		for (j = 27; j < numLandmarks; j++) {
			x1 = current_shape.at<float>(j);
			y1 = current_shape.at<float>(j + numLandmarks);
			LBPTar[j - 27] = (float*)calloc(bin, sizeof(float));

			for (y2 = y1 - heightblk / 2; y2 < y1 + heightblk / 2; y2++) {
				for (x2 = x1 - widhtblk / 2; x2 < x1 + widhtblk / 2; x2++) {
					imgTemp.at<uchar>((y2 - y1) + heightblk / 2, (x2 - x1) + widhtblk / 2) = imgGray.at<uchar>(y2, x2);
					LandmarkLBPTar[j - 27] = GetLBP(imgTemp, 3);
				}
			}
			LBPTar[j - 27] = GetLBPHisto(LandmarkLBPTar[j - 27], bin);
		}

		corelation = GetSimilarity(HOGRef, LBPTar, ENM, bin, COSINE);
		std::printf("corelation : %f\n", corelation);
		if (corelation > 0.95) {
			for (j = 27; j < numLandmarks; j++) {
				x1 = current_shape.at<float>(j);
				y1 = current_shape.at<float>(j + numLandmarks);

				circle(img, Point(x1, y1), 2, Scalar(0, 255, 0), -1);
			}
		}
		else {
			for (j = 27; j < numLandmarks; j++) {
				x1 = current_shape.at<float>(j);
				y1 = current_shape.at<float>(j + numLandmarks);

				circle(img, Point(x1, y1), 2, Scalar(0, 0, 255), -1);
			}
		}

		freePtr(LBPTar, ENM);

		cv::imshow("img", img);
		if (waitKey(30) >= 0) {
			caputure.release();
			destroyAllWindows();
			break;
		}
	}

	freePtr(HOGRef, ENM);

	return 0;
}