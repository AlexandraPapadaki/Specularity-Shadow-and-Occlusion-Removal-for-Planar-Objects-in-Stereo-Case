#include <iostream>
#include <iomanip>

#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

using namespace std;
using namespace cv;

static class UTIL{

public:

	static Mat makeLabelBorders(Mat &labels, int lines = 1)
	{	
		if((labels.cols*labels.rows) > (1000*1000)) lines = 2;
		if ((labels.cols*labels.rows) > (1500*1500)) lines = 3;

		Mat borders(labels.size(), CV_8U, Scalar(255));
		for (int y = 0; y < (labels.rows - 1); y++)
			for (int x = 0; x < (labels.cols - 1); x++) {
				int l = labels.at<int>(y, x);
				int lx = labels.at<int>(y, x + 1);
				int ly = labels.at<int>(y + 1, x);

				if (l != lx) {
					borders.at<uchar>(y, x + 1) = 0;
					if(lines > 1)
						borders.at<uchar>(y, x) = 0;
					if (lines > 2)
						borders.at<uchar>(y, x + 2) = 0;
				}
				if (l != ly) {
					borders.at<uchar>(y + 1, x) = 0;
					if (lines > 1)
						borders.at<uchar>(y, x) = 0;
					if (lines > 2)
						borders.at<uchar>(y + 2, x) = 0;
				}
			}
		return borders;
	}

	static Mat drawLabelBorders(Mat &labels, Mat &img, Scalar color = Scalar(0, 0, 0), int lines = 1) {
		Mat borders = makeLabelBorders(labels, lines);
		Mat draw = Mat(img.size(), CV_8UC3, color);
		img.copyTo(draw, borders);
		return draw;
	}

};
