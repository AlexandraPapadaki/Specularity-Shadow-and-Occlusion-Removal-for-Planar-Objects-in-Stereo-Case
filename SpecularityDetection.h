#pragma once

#include <iostream>
#include <thread>

#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

#include "App.h"
#include "ImageSegmentation.h"


/*	Class which provides the means to detect specularities
	and shadows based on image segmentation
*/
class SpecularityDetection
{	
public:	
	Mat m_tarImg;
	Mat m_srcImg;
	Mat m_observable;

	bool m_bTarHasShadows;
	bool m_bSrcHasShadows;

	Mat m_tarShadows;
	Mat m_srcShadows;
	Mat m_specularRegions;

	// source image segmentation inherited from target image segmentation
	ImageSegmentation* m_pTarSrcSegmentation;
	// target image segmentation inherited from source image segmentation
	ImageSegmentation* m_pSrcTarSegmentation;

	void Initialize(Mat &tarImg, Mat &srcImg, Mat &observable,
					bool tarHasShadows, bool srcHasShadows);
	double GetInitialThreshold();

	void DetectSpecularities(float distTh);
	void DetectTargetShadows(App::ShadowDetectionContext& ctx) {
		DetectShadows(m_pTarSrcSegmentation, ctx);
		m_tarShadows = ctx.shadows;
	}
	void DetectSourceShadows(App::ShadowDetectionContext& ctx) {
		DetectShadows(m_pSrcTarSegmentation, ctx);
		m_srcShadows = ctx.shadows;
	}
	Mat MakeCorrectionRegion();

private:
	void DetectShadows(ImageSegmentation* pSeg, App::ShadowDetectionContext& ctx);
};

