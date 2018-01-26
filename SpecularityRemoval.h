#pragma once
#include "App.h"
#include "SpecularityDetection.h"

#include <iostream>

#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include "opencv2\photo.hpp"


/*	This class manages the correction of the image content
	from the correction region which has been generated
	from the masks of the detection results
*/
class SpecularityRemoval
{
public:
	SpecularityRemoval(SpecularityDetection* pDet);

	Mat m_tarImg, m_srcImg,
		m_areaToCorrect;

	SpecularityDetection* m_pDet;

	Mat MakeCorrectedImage();
	Mat MakeCorrectedImage(Mat& blendingBorder, Mat& correctionArea);
private:
	void RemoveSmallComponents();
	Mat MakeBorderBlendingMask();
	Mat MakeBorderBlendingMask(Size kernelSize, int k);
}; 

