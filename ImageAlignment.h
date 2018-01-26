#pragma once

#include <iostream>

#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\calib3d.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <opencv2\video.hpp>

#include "App.h"

/*	Class that provides the functionality for object
	rectification and image alignment
*/
class ImageAlignment
{
public:
	Mat m_targetImg, m_warpedImg; 
	Mat m_observable;

	void RectifyObject(	Mat &img, vector<Point2i> sourcePts, 
						vector<Point2i> targetPts);
	void Align(Mat &target, Mat &source);
	
	Mat ObservableArea(Mat &img){
		Mat output;
		img.copyTo(output, m_observable);
		return output;
	}
};

