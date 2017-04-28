#include<iostream>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<time.h>

using namespace std;
using namespace cv;

void edgeDetect(Mat src, Mat& dst);
void phi(Mat img, Mat& grd);
void quantification(Mat src, Mat& dst, Mat grd, int w);
void imgMerge(Mat imgEdge, Mat imgColor, Mat& dst);

int main(){
	clock_t start_t, end_t;
	start_t = clock();
	Mat img;
	img = imread("E:\\file\\lena.jpg");
	if (!img.data){
		cout << "img load failed" << endl;
		waitKey(0);
		return -1;
	}
	cout << "read done" << endl;

	namedWindow("Original");
	imshow("Original", img);
	/*----smoth----*/
	int d = 30;//邻域直径
	double sigmaColor = 30;//输入的可调节参数
	double sigmaSpace = 40;//输入的可调节参数
	Mat imBil, temp;
	bilateralFilter(img, temp, d, sigmaColor, sigmaSpace);
	bilateralFilter(temp, imBil, d, sigmaColor, sigmaSpace);

	int h = img.size().height;
	int w = img.size().width;

	Mat imgEdge;
	edgeDetect(imBil, imgEdge);
	//edge--CV_8UC1(Gray)

	/*----magnitude----*/
	Mat grd = Mat(h, w, CV_8UC1);
	phi(img, grd);
	//grd---CV_8UC1(Gray)

	/*----quantification----*/
	Mat imgColor;
	int width = 30;//输入的可调节参数
	quantification(imBil, imgColor, grd, width);
	//imColor---CV_8UC3(RGB)

	/*----merge----*/
	Mat fin, fin2;
	imgMerge(imgEdge, imgColor, fin);
	//fin.convertTo(fin2, CV_8UC3, 255.0);

	/////////////////////////////这一位置是否要进行平滑待定
	medianBlur(fin, fin2, 3);
	namedWindow("fin");
	imshow("fin", fin2);
	imwrite("fin.jpg", fin);

	end_t = clock();
	cout << "total time :" << (double)(end_t - start_t) / CLOCKS_PER_SEC << "s" << endl;
	waitKey(0);
	return 0;
}

void edgeDetect(Mat src, Mat& dst){
	/*src--CV_8UC3 RGB image after smooth(RGB)
	dst---CV_8UC1(Gray)
	*/
	int h = src.size().height;
	int w = src.size().width;

	Mat temp;

	/*----split----*/
	Mat imLab;
	Mat Lab[3];
	cvtColor(src, imLab, COLOR_RGB2Lab);
	split(imLab, Lab);

	/*----DoG----*/
	Mat imEdge;
	/*GaussianBlur(Lab[0], temp, Size(5, 5), 0, 0);
	GaussianBlur(Lab[0], imEdge, Size(9, 9), 0, 0);*/
	GaussianBlur(Lab[0], temp, Size(3, 3), 0, 0);
	GaussianBlur(temp, imEdge, Size(9, 9), 0, 0);
	imEdge = imEdge - temp;

	/*----threshold----*/
	double maxVal;
	minMaxLoc(imEdge, 0, &maxVal, 0, 0);
	int th = 15;//试验多次选合适的值
	int p = 0;
	double n = 255.0 / maxVal;
	for (int row = 0; row < src.rows; row++){
		for (int col = 0; col< src.cols; col++){
			p = imEdge.at<uchar>(row, col);
			p = p*n;
			if (p > th){
				imEdge.at<uchar>(row, col) = p;
			}
			else{
				imEdge.at<uchar>(row, col) = 0;
			}
		}
	}

	/*----final edge----*/
	//medianBlur(imEdge, dst, 3);
	equalizeHist(imEdge, dst);
	namedWindow("edge");
	imshow("edge", dst);

	imwrite("edge.jpg", dst);
}

void phi(Mat img, Mat& grd){
	/*img---CV_8UC3(RGB)----original image(before smooth)
	grd---CV_8UC1(Gray)*/
	Mat src, temp;
	cvtColor(img, src, COLOR_RGB2GRAY);

	Mat sobx, soby, sob;
	Sobel(src, sobx, CV_32FC1, 1, 0, 3);
	Sobel(src, soby, CV_32FC1, 0, 1, 3);

	magnitude(sobx, soby, sob);
	convertScaleAbs(sob, temp);
	double maxVal;
	minMaxLoc(temp, 0, &maxVal);
	temp.convertTo(grd, CV_8UC1, 255.0 / maxVal);
}

void quantification(Mat src, Mat& dst, Mat grd, int w){
	/*src---CV_8UC3(RGB)
	grd---CV_8UC1(Gray)
	det---CV_8UC3(RGB)*/

	Mat temp;//Lab image of src
	cvtColor(src, temp, COLOR_RGB2Lab);

	int q, n, g;
	for (int row = 0; row < src.rows; row++){
		for (int col = 0; col< src.cols; col++){
			q = temp.at<Vec3b>(row, col)[0];
			n = (q / w) * w;
			g = grd.at<uchar>(row, col);
			int d = n + w - q;
			if (d < w / 2){
				n += w;
			}
			temp.at<Vec3b>(row, col)[0] = n + (w / 2)*tanh(g*(q - n) / 40);
		}
	}
	cvtColor(temp, dst, COLOR_Lab2RGB);
}

void imgMerge(Mat imgEdge, Mat imgColor, Mat& dst){
	/*imgEdge---CV_8UC1(Gray)
	imgColor---CV_8UC3(RGB)*/

	/*----method 1----*/
	/*Mat edge;
	imgEdge = 255.0 - imgEdge;
	imgEdge.convertTo(edge, CV_32FC3, 1.0 / 255.0);
	Mat color;
	imgColor.convertTo(color, CV_32FC3, 1.0 / 255.0);

	Mat temp[3];
	split(color, temp);
	multiply(temp[0], edge, temp[0]);
	multiply(temp[1], edge, temp[1]);
	multiply(temp[2], edge, temp[2]);

	merge(temp, 3, dst);*/

	/*----method 2----*/
	Mat img_Lab, Lab[3];
	cvtColor(imgColor, img_Lab, COLOR_RGB2Lab);
	split(img_Lab, Lab);

	int h = imgColor.size().height;
	int w = imgColor.size().width;
	Mat edge = Mat(h, w, CV_32FC1);
	edge = (255.0 - imgEdge) / 255.0;

	Mat temp;
	Lab[0] = Lab[0] - imgEdge;
	//multiply(Lab[0], edge, Lab[0]);
	//equalizeHist(temp, Lab[0]);
	merge(Lab, 3, img_Lab);
	cvtColor(img_Lab, dst, COLOR_Lab2RGB);
}