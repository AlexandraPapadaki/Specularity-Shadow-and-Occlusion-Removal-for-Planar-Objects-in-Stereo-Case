#pragma once

#include <iostream>

#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\ximgproc.hpp>
#include "utility.h"
#include "App.h"

using namespace std;
using namespace cv;

/*	Class which provides the functionality for image segmentation,
	ImageSegmentation is a subclass of SpecularityDetection
*/
class ImageSegmentation
{
public:
	// images
	Mat m_img1, m_img2, 
		m_observable;

	// segments
	Mat m_labels;
	int m_numLabels;
	Mat m_segSizes;

	// segment description
	Mat m_img1Values, m_img2Values,
		m_img1ValuesGRAY, m_img2ValuesGRAY,
		m_img1IntensitiesBGR, m_img2IntensitiesBGR,
		m_intensitiesDiffBGR;

	// regions
	Mat m_regions;
	int m_numRegions;
	Mat m_regionSizes;

	// region description
	vector<Mat> m_intermediateRegions;
	// 0-BGR, 1-LAB values
	Mat m_img1RegionValues, m_img2RegionValues,
		m_regionDistBGR, m_regionDistLAB;
	Mat m_mapping;
	vector<vector<int>> m_borderSegments;
	Mat m_isBorderSegment;
	vector<vector<int>> m_borderNeighborhood;

	void Segmentation(Mat &img1, Mat &img2, Mat &observable, int numSegments, int iterations);
	void MakeRegions(int minSize);
	void GetNeighborhood(Mat &labels, Mat &segmentsOfInterest, vector<vector<int>> &neighborhood);

private:
	void DescribeSegments();
	void DescribeIntermediateRegions(Mat& regions, int numRegions, Mat& regionSizes, Mat& regionValues);
	int RemapLabels(Mat &labels, Mat &observable, int numLabels);
	int MergeToRegions(Mat& regions, Mat& labels, int numLabels, Mat& imgValues, double distTh);
	int AssignSegmentsToRegions(Mat& regions, Mat& regionValues, int sizeTh, Mat& finalRegions);
	void MakeBorderSegments();

};

