#include "SpecularityDetection.h"


/*	Start image segmentation, if shadow detection is 
	desired bigger regions have to be formed, otherwise 
	skip region segmentation to make the algorithm run 
	faster, since in this case regions are superfluous
*/
void SpecularityDetection::Initialize(
	Mat &tarImg, Mat &srcImg, Mat &observable, 
	bool tarHasShadows, bool srcHasShadows)
{
	m_tarImg = tarImg.clone();
	m_srcImg = srcImg.clone();
	m_observable = observable.clone();
	// regions are not needed if shadow detection is disabled 
	m_bTarHasShadows = tarHasShadows;
	m_bSrcHasShadows = srcHasShadows;
	m_tarShadows = Mat(m_srcImg.size(), CV_8U, Scalar(0));
	m_srcShadows = Mat(m_srcImg.size(), CV_8U, Scalar(0));

	// start image segmentation
	int minRegionSize = (m_tarImg.cols * m_tarImg.rows) * MIN_REGION_SIZE_FRACTION;

	// do this in parallel
	thread thread([this, minRegionSize]() 
	{
		m_pSrcTarSegmentation = new ImageSegmentation();
		m_pSrcTarSegmentation->Segmentation(m_srcImg, m_tarImg,	m_observable, 
											MAX_NUM_SEGMENTS, SEED_ITERATIONS);
		if(m_bSrcHasShadows)
			m_pSrcTarSegmentation->MakeRegions(minRegionSize);
	});
	
	m_pTarSrcSegmentation = new ImageSegmentation();
	m_pTarSrcSegmentation->Segmentation(m_tarImg, m_srcImg, m_observable,
										MAX_NUM_SEGMENTS, SEED_ITERATIONS);
	if(tarHasShadows)
		m_pTarSrcSegmentation->MakeRegions(minRegionSize);

	thread.join();
}

/*	Assemble the final correction mask from all the 
	detection results, shadows detected in the source
	image are removed from the region since they cause
	the detection of fake specularities
*/
Mat SpecularityDetection::MakeCorrectionRegion()
{
	Mat correctionRegion;
	Mat tarShadowMask = m_tarShadows.clone(),
		srcShadowMask = m_srcShadows.clone();

	Size kernelSize = Size(3, 3);

	// dilate shadow regions to avoid blending them back into image  
	Mat kernel = getStructuringElement(MORPH_CROSS, kernelSize);
	dilate(srcShadowMask, srcShadowMask, kernel, Point(-1, -1), SHADOW_DILATION_ITERATIONS);
	dilate(tarShadowMask, tarShadowMask, kernel, Point(-1, -1), SHADOW_DILATION_ITERATIONS);
	// specular region without source image shadows + target image shadows
	correctionRegion = (m_specularRegions & (255 - srcShadowMask)) | tarShadowMask;

	// smooth rough contours due to segments which will cause altering of blending border
	dilate(correctionRegion, correctionRegion, kernel, Point(-1, -1), BORDER_SMOOTHING_ITERATIONS);

	return correctionRegion;
}

/*	Get an initial guess of an initial threshold
	by calculating the average inensity difference 
	of the corresponding segments in both images
*/
double SpecularityDetection::GetInitialThreshold()
{
	App::ConsolePrint("Computing initial threshold");

	ImageSegmentation* pTarSeg = m_pTarSrcSegmentation;
	int numLabels = pTarSeg->m_numLabels;

	// exclude segments in shadows or occlusions
	bool* isShadowSeg = new bool[numLabels];
	fill(isShadowSeg, isShadowSeg + numLabels, 0);
	for (int y = 0; y < m_tarShadows.rows; y++)
		for (int x = 0; x < m_tarShadows.cols; x++) {
			int l = pTarSeg->m_labels.at<int>(y, x);
			if (m_tarShadows.at<uchar>(y, x) || m_srcShadows.at<uchar>(y, x))
				isShadowSeg[l] = 1;
		}


	vector<double> dist_BGR, dist_LAB, diff_L, diff_GRAY;

	// average distance
	double	avg_diff_BGR = 0, avg_dist_BGR = 0, avg_dist_LAB = 0, avg_diff_L = 0, avg_diff_GRAY = 0,
			med_dist_BGR = 0, med_dist_LAB = 0, med_diff_L = 0, med_diff_GRAY = 0;

	int iterations = numLabels;
	for (int l = 0; l < iterations; l++) {
		if (isShadowSeg[l]) {
			numLabels--;
			continue;
		}

		avg_diff_BGR += pTarSeg->m_intensitiesDiffBGR.at<float>(l, 0);
		Vec3d dRGBVec = (Vec3d)pTarSeg->m_img1Values.at<Vec3i>(l, 0) - (Vec3d)pTarSeg->m_img2Values.at<Vec3i>(l, 0);
		double dRGB = sqrt(dRGBVec[0] * dRGBVec[0] + dRGBVec[1] * dRGBVec[1] + dRGBVec[2] * dRGBVec[2]);
		avg_dist_BGR += dRGB;
		Vec3d dLABVec = (Vec3d)pTarSeg->m_img1Values.at<Vec3i>(l, 3) - (Vec3d)pTarSeg->m_img2Values.at<Vec3i>(l, 3);
		double dLAB = sqrt(dLABVec[0] * dLABVec[0] + dLABVec[1] * dLABVec[1] + dLABVec[2] * dLABVec[2]);
		avg_dist_LAB += dLAB;
		double dL = abs(dLABVec[0]);
		avg_diff_L += dL;
		double dGRAY = abs(pTarSeg->m_img1ValuesGRAY.at<int>(l, 0) - pTarSeg->m_img2ValuesGRAY.at<int>(l, 0));
		avg_diff_GRAY += dGRAY;

		dist_BGR.push_back(dRGB);
		dist_LAB.push_back(dLAB);
		diff_L.push_back(dL);
		diff_GRAY.push_back(dGRAY);
	}
	avg_diff_BGR /= numLabels;
	avg_dist_BGR /= numLabels;
	avg_dist_LAB /= numLabels;
	avg_diff_L /= numLabels;
	avg_diff_GRAY /= numLabels;
		
	// median distance
	sort(dist_BGR.begin(), dist_BGR.end());
	sort(dist_LAB.begin(), dist_LAB.end());
	sort(diff_L.begin(), diff_L.end());
	sort(diff_GRAY.begin(), diff_GRAY.end());
	med_dist_BGR = dist_BGR[numLabels / 2] / numLabels;
	med_dist_LAB = dist_LAB[numLabels / 2] / numLabels;
	med_diff_L = diff_L[numLabels / 2] / numLabels;
	med_diff_GRAY = diff_GRAY[numLabels / 2] / numLabels;

	// average distance of highest distance segments
	double th_BGR = 0, th_LAB = 0, th_L = 0, th_GRAY = 0;
	int upper = INITIAL_TH_SAMPLES_PERCENTAGE*(double)numLabels;
	for (int l = upper; l < numLabels; l++) {
		th_BGR += dist_BGR[l];
		th_LAB += dist_LAB[l];
		th_L += diff_L[l];
		th_GRAY += diff_GRAY[l];
	}
	
	th_BGR = th_BGR / (numLabels - upper) * INITIAL_TH_FRACTION;
	th_LAB = th_LAB / (numLabels - upper) * INITIAL_TH_FRACTION;
	th_L = th_L / (numLabels - upper) * INITIAL_TH_FRACTION;
	th_GRAY = th_GRAY / (numLabels - upper) * INITIAL_TH_FRACTION;

	//App::ConsolePrint("INITIAL THRESHOLDS: THRESH | AVG | MEDIAN");
	//App::ConsolePrint("dist_BGR :  " + to_string(th_BGR) + " | " + to_string(avg_dist_BGR) + " | " + to_string(med_dist_BGR));
	//App::ConsolePrint("dist_LAB :  " + to_string(th_LAB) + " | " + to_string(avg_dist_LAB) + " | " + to_string(med_dist_LAB));
	//App::ConsolePrint("diff_L :    " + to_string(th_L) + " | " + to_string(avg_diff_L) + " | " + to_string(med_diff_L));
	//App::ConsolePrint("diff_GRAY : " + to_string(th_GRAY) + " | " + to_string(avg_diff_GRAY) + " | " + to_string(med_diff_GRAY));
	App::ConsolePrint("INITIAL THRESHOLD: " + to_string(th_GRAY));
	double thresh;
	if (App::getScSpec() == App::INTENSITY_DIFF_RGB) thresh = avg_diff_BGR;
	else if (App::getScSpec() == App::DIST_RGB) thresh = th_BGR;
	else if (App::getScSpec() == App::DIST_LAB) thresh = th_LAB;
	else if (App::getScSpec() == App::L_DIFF_LAB) thresh = th_L;
	else if (App::getScSpec() == App::GRAY_DIFF) thresh = th_GRAY;

	return thresh;
}

/*	Find the corresponding segments with a difference in intensity
	above a threshold distTh and these segments to the specular region
*/
void SpecularityDetection::DetectSpecularities(float distTh)
{
	Mat labels = m_pTarSrcSegmentation->m_labels;
	Mat img1Values = m_pTarSrcSegmentation->m_img1Values;
	Mat img2Values = m_pTarSrcSegmentation->m_img2Values;
	Mat img1ValuesGRAY = m_pTarSrcSegmentation->m_img1ValuesGRAY;
	Mat img2ValuesGRAY = m_pTarSrcSegmentation->m_img2ValuesGRAY;

	m_specularRegions = Mat(labels.size(), CV_8U, Scalar(0));

	// previous approach
	if(App::getScSpec() == App::INTENSITY_DIFF_RGB){
		double min, max, norm;
		minMaxIdx(m_pTarSrcSegmentation->m_intensitiesDiffBGR, &min, &max);
		norm = 255.0 / cv::max(abs(min), max);
		Mat normIntensities = norm*m_pTarSrcSegmentation->m_intensitiesDiffBGR;

		for (int y = 0; y < labels.rows; y++)
			for (int x = 0; x < labels.cols; x++) {
				if ((normIntensities.at<float>(labels.at<int>(y, x), 0) * norm) > distTh)
					m_specularRegions.at<uchar>(y, x) = 255;
			}
		return;
	}

	for (int y = 0; y < labels.rows; y++)
		for (int x = 0; x < labels.cols; x++) {
			int l = labels.at<int>(y, x);
			bool spec = false;

			if (App::getScSpec() == App::DIST_RGB) {
				Vec3d dRGB = (Vec3d)img1Values.at<Vec3i>(l, 0)- (Vec3d)img2Values.at<Vec3i>(l, 0);
				spec = ((img1ValuesGRAY.at<int>(l, 0) - img2ValuesGRAY.at<int>(l, 0)) > 0) &&
						(sqrt(dRGB[0] * dRGB[0] + dRGB[1] * dRGB[1] + dRGB[2] * dRGB[2]) > distTh);
			}
			else if (App::getScSpec() == App::DIST_LAB) {
				Vec3d dLAB = (Vec3d)img1Values.at<Vec3i>(l, 3) - (Vec3d)img2Values.at<Vec3i>(l, 3);
				spec = (dLAB[0] > 0) && (sqrt(dLAB[0] * dLAB[0] + dLAB[1] * dLAB[1] + dLAB[2] * dLAB[2]) > distTh);
			}
			else if (App::getScSpec() == App::L_DIFF_LAB) {
				spec = (img1Values.at<Vec3i>(l, 3)[0] - img2Values.at<Vec3i>(l, 3)[0]) > distTh;
			}
			else if (App::getScSpec() == App::GRAY_DIFF) {
				spec = (img1ValuesGRAY.at<int>(l, 0) - img2ValuesGRAY.at<int>(l, 0)) > distTh;
			}


			if (spec) m_specularRegions.at<uchar>(y, x) = 255;
		}


}

/*	Detect shadows in the image, for both images compute the highest 
	"gradient" of a border segment's intensity to the neighboring  
	border segments of the opposite region and filter all the segments 
	by a threshold "gradTh", if after filtering the segment is present
	in one but not the other image it casts a vote for it's region to
	be a possible candidate for a shadow region  
*/
void SpecularityDetection::DetectShadows(
	ImageSegmentation* pSeg, App::ShadowDetectionContext& ctx)
{
	//cout << endl << "Detect Shadows: " << endl;

	int numLabels = pSeg->m_numLabels;
	int numRegions = pSeg->m_numRegions;
	Mat labels = pSeg->m_labels;
	vector<vector<int>> neighborhood
		= pSeg->m_borderNeighborhood;

	double gradTh = (double)ctx.gradTh;
	double minPercentageDetected = max(10.0, (double)ctx.minPercentageDetected) / 100;
	double minInitially = max(10.0, (double)ctx.minInitiallyDetected) / 100;

	// check border segments for gradients at region border in both images
	Mat img1Values = pSeg->m_img1Values;
	Mat img2Values = pSeg->m_img2Values;
	Mat img1ValuesGRAY = pSeg->m_img1ValuesGRAY;
	Mat img2ValuesGRAY = pSeg->m_img2ValuesGRAY;

	bool* img1StrongGrad = new bool[numLabels];
	bool* img2StrongGrad = new bool[numLabels];
	fill(img1StrongGrad, img1StrongGrad + numLabels, 0);
	fill(img2StrongGrad, img2StrongGrad + numLabels, 0);
	for (int l = 1; l < numLabels; l++) {
		if (!pSeg->m_isBorderSegment.at<uchar>(l, 0)) continue;

		vector<int> neighbors = neighborhood[l];
		for (int n : neighbors) {
			// check for strong positive "gradient" of the border segment in both images
			if (App::getScShadows() == App::INTENSITY_DIFF_RGB) {
				if ((pSeg->m_img1IntensitiesBGR.at<float>(n, 0) - pSeg->m_img1IntensitiesBGR.at<float>(l, 0)) > gradTh)
					img1StrongGrad[l] = 1;
				if ((pSeg->m_img2IntensitiesBGR.at<float>(n, 0) - pSeg->m_img2IntensitiesBGR.at<float>(l, 0)) > gradTh)
					img2StrongGrad[l] = 1;
			}
			else if (App::getScShadows() == App::DIST_RGB) {
				Vec3d dRGB1 = (Vec3d)img1Values.at<Vec3i>(n, 0) - (Vec3d)img1Values.at<Vec3i>(l, 0);
				Vec3d dRGB2 = (Vec3d)img2Values.at<Vec3i>(n, 0) - (Vec3d)img2Values.at<Vec3i>(l, 0);
				if (sqrt(dRGB1[0] * dRGB1[0] + dRGB1[1] * dRGB1[1] + dRGB1[2] * dRGB1[2]) > gradTh)
					img1StrongGrad[l] = 1;
				if (sqrt(dRGB2[0] * dRGB2[0] + dRGB2[1] * dRGB2[1] + dRGB2[2] * dRGB2[2]) > gradTh)
					img2StrongGrad[l] = 1;
			}
			else if (App::getScShadows() == App::DIST_LAB) {
				Vec3d dLAB1 = (Vec3d)img1Values.at<Vec3i>(n, 3) - (Vec3d)img1Values.at<Vec3i>(l, 3);
				Vec3d dLAB2 = (Vec3d)img2Values.at<Vec3i>(n, 3) - (Vec3d)img2Values.at<Vec3i>(l, 3);
				if (sqrt(dLAB1[0] * dLAB1[0] + dLAB1[1] * dLAB1[1] + dLAB1[2] * dLAB1[2]) > gradTh)
					img1StrongGrad[l] = 1;
				if (sqrt(dLAB2[0] * dLAB2[0] + dLAB2[1] * dLAB2[1] + dLAB2[2] * dLAB2[2]) > gradTh)
					img2StrongGrad[l] = 1;
			}
			else if (App::getScShadows() == App::L_DIFF_LAB) {
				if ((img1Values.at<Vec3i>(n, 3)[0] - img1Values.at<Vec3i>(l, 3)[0]) > gradTh)
					img1StrongGrad[l] = 1;
				if ((img2Values.at<Vec3i>(n, 3)[0] - img2Values.at<Vec3i>(l, 3)[0]) > gradTh)
					img2StrongGrad[l] = 1;
			}
			else if (App::getScShadows() == App::GRAY_DIFF) {
				if ((img1ValuesGRAY.at<int>(n, 0) - img1ValuesGRAY.at<int>(l, 0)) > gradTh)
					img1StrongGrad[l] = 1;
				if ((img2ValuesGRAY.at<int>(n, 0) - img2ValuesGRAY.at<int>(l, 0)) > gradTh)
					img2StrongGrad[l] = 1;
			}
		}
	}

	// get regions which are candidates for shadows  
	int* numStrongGrads = new int[numRegions];
	fill(numStrongGrads, numStrongGrads + numRegions, 0);
	for (int r = 1; r < numRegions; r++)
		for (int l : pSeg->m_borderSegments[r]) {
			// strong gradient in one image that does not occure 
			// in the other image is a hint for a shadow
			if (img1StrongGrad[l] && !img2StrongGrad[l]) {
				numStrongGrads[r]++;
			}
		}

	/// ...
	bool* isShadow = new bool[numRegions];
	fill(isShadow, isShadow + numRegions, 0);
	bool* assumeShadow = new bool[numRegions];
	fill(assumeShadow, assumeShadow + numRegions, 0);
	int* numSegsAtBorderToShadow = new int[numRegions];
	fill(numSegsAtBorderToShadow, numSegsAtBorderToShadow + numRegions, 0);
	int* votingTrueBorderSegs = new int[numRegions];
	for (int r = 1; r < numRegions; r++) {
		votingTrueBorderSegs[r] = numStrongGrads[r];
	}

	while (1) {
		int newDetections = 0;

		for (int r = 1; r < numRegions; r++) {
			// check if region contains a sufficient 
			// number of candidate border segments 
			float numBorderSegs = (float)pSeg->m_borderSegments[r].size() - numSegsAtBorderToShadow[r];
			float percInitiallyDetected = (float)votingTrueBorderSegs[r] / numBorderSegs;
			if (percInitiallyDetected > minInitially){
				if (!assumeShadow[r]) newDetections++;
				assumeShadow[r] = 1;
			}
		}

		// for each shadow region candidate get the number of 
		// segments with strong gradients as well as the total 
		// number of segments at the borders to non shadow regions,
		// this is necessary to detect shadows which are split into
		// multiple regions
		fill(numSegsAtBorderToShadow, numSegsAtBorderToShadow + numRegions, 0);
		fill(votingTrueBorderSegs, votingTrueBorderSegs + numRegions, 0);
		for (int r = 1; r < numRegions; r++) {
			// skip if region is not a candidate for shadows
			//if (!assumeShadow[r]) continue;

			for (int l : pSeg->m_borderSegments[r]) {
				bool atBorderToShadow = false;
				bool strongGrad = false;
				for (int n : neighborhood[l]) {
					// neighbor is assumed to belong to a shadow region
					if (assumeShadow[pSeg->m_mapping.at<int>(n, 0)]) {
						atBorderToShadow = true;
						continue;
					}

					// neighbor is a true border
					if (img1StrongGrad[l] && !img2StrongGrad[l])
						strongGrad = true;
				}

				if (atBorderToShadow) numSegsAtBorderToShadow[r]++;
				else if (strongGrad) votingTrueBorderSegs[r]++;
			}
		}

		//check if region is surrounded by shadows -> it most likely belongs to the shadow
		const float minPercSurroundedByShadow = 1.0;
		for (int r = 1; r < numRegions; r++) {
			float numBorderSegs = (float)pSeg->m_borderSegments[r].size();
			float det = (float)(votingTrueBorderSegs[r] + numSegsAtBorderToShadow[r]);
			if ((det / numBorderSegs) > minPercSurroundedByShadow || numSegsAtBorderToShadow[r] == numBorderSegs) {
				if (!assumeShadow[r]) newDetections++;
				assumeShadow[r] = 1;	
			}
		}

		//cout << "New detections in this iteration: " << to_string(newDetections) << endl;
		if (newDetections == 0) break;

		// TODO: iteration probably not necessary
		//break;
	}

	ctx.shadows = Mat(labels.size(), CV_8U, Scalar(0));
	ctx.diffEdges = Mat(labels.size(), CV_8U, Scalar(0));
	for (int y = 0; y < labels.rows; y++)
		for (int x = 0; x < labels.cols; x++) {

			if (img1StrongGrad[labels.at<int>(y, x)]
				&& !img2StrongGrad[labels.at<int>(y, x)])
				ctx.diffEdges.at<uchar>(y, x) = 255;

			if (assumeShadow[pSeg->m_regions.at<int>(y, x)])
				ctx.shadows.at<uchar>(y, x) = 255;
		}

	delete[] img1StrongGrad, img2StrongGrad, numStrongGrads, assumeShadow, isShadow, votingTrueBorderSegs, numSegsAtBorderToShadow;
}



void _deprecated_DetectShadows(
	 ImageSegmentation* pSeg, App::ShadowDetectionContext& ctx)
{
	int numLabels = pSeg->m_numLabels;
	int numRegions = pSeg->m_numRegions;
	Mat labels = pSeg->m_labels;
	vector<vector<int>> neighborhood
		= pSeg->m_borderNeighborhood;
	
	double gradTh = (double)ctx.gradTh;
	double minPercentageDetected = max(10.0, (double)ctx.minPercentageDetected) / 100;
	double minInitially = max(10.0, (double)ctx.minInitiallyDetected) / 100;
	
	// check border segments for gradients at region border in both images
	Mat img1Values = pSeg->m_img1Values;
	Mat img2Values = pSeg->m_img2Values;
	Mat img1ValuesGRAY = pSeg->m_img1ValuesGRAY;
	Mat img2ValuesGRAY = pSeg->m_img2ValuesGRAY;

	bool* img1StrongGrad = new bool[numLabels];
	bool* img2StrongGrad = new bool[numLabels];
	fill(img1StrongGrad, img1StrongGrad + numLabels, 0);
	fill(img2StrongGrad, img2StrongGrad + numLabels, 0);
	for (int l = 1; l < numLabels; l++) {
		if (!pSeg->m_isBorderSegment.at<uchar>(l, 0)) continue;

		vector<int> neighbors = neighborhood[l];
		for (int n : neighbors) {
			// check for strong positive "gradient" of the border segment in both images
			if (App::getScShadows() == App::INTENSITY_DIFF_RGB) {
				if ((pSeg->m_img1IntensitiesBGR.at<float>(n, 0) - pSeg->m_img1IntensitiesBGR.at<float>(l, 0)) > gradTh)
					img1StrongGrad[l] = 1;
				if ((pSeg->m_img2IntensitiesBGR.at<float>(n, 0) - pSeg->m_img2IntensitiesBGR.at<float>(l, 0)) > gradTh)
					img2StrongGrad[l] = 1;
			}
			else if (App::getScShadows() == App::DIST_RGB) {
				Vec3d dRGB1 = (Vec3d)img1Values.at<Vec3i>(n, 0) - (Vec3d)img1Values.at<Vec3i>(l, 0);
				Vec3d dRGB2 = (Vec3d)img2Values.at<Vec3i>(n, 0) - (Vec3d)img2Values.at<Vec3i>(l, 0);
				if (sqrt(dRGB1[0] * dRGB1[0] + dRGB1[1] * dRGB1[1] + dRGB1[2] * dRGB1[2]) > gradTh)
					img1StrongGrad[l] = 1;
				if (sqrt(dRGB2[0] * dRGB2[0] + dRGB2[1] * dRGB2[1] + dRGB2[2] * dRGB2[2]) > gradTh)
					img2StrongGrad[l] = 1;
			}
			else if (App::getScShadows() == App::DIST_LAB) {
				Vec3d dLAB1 = (Vec3d)img1Values.at<Vec3i>(n, 3) - (Vec3d)img1Values.at<Vec3i>(l, 3);
				Vec3d dLAB2 = (Vec3d)img2Values.at<Vec3i>(n, 3) - (Vec3d)img2Values.at<Vec3i>(l, 3);
				if (sqrt(dLAB1[0] * dLAB1[0] + dLAB1[1] * dLAB1[1] + dLAB1[2] * dLAB1[2]) > gradTh)
					img1StrongGrad[l] = 1;
				if (sqrt(dLAB2[0] * dLAB2[0] + dLAB2[1] * dLAB2[1] + dLAB2[2] * dLAB2[2]) > gradTh)
					img2StrongGrad[l] = 1;
			}
			else if (App::getScShadows() == App::L_DIFF_LAB) {
				if ((img1Values.at<Vec3i>(n, 3)[0] - img1Values.at<Vec3i>(l, 3)[0]) > gradTh)
					img1StrongGrad[l] = 1;
				if ((img2Values.at<Vec3i>(n, 3)[0] - img2Values.at<Vec3i>(l, 3)[0]) > gradTh)
					img2StrongGrad[l] = 1;
			}
			else if (App::getScShadows() == App::GRAY_DIFF) {
				if ((img1ValuesGRAY.at<int>(n, 0) - img1ValuesGRAY.at<int>(l, 0)) > gradTh)
					img1StrongGrad[l] = 1;
				if ((img2ValuesGRAY.at<int>(n, 0) - img2ValuesGRAY.at<int>(l, 0)) > gradTh)
					img2StrongGrad[l] = 1;
			}
		}
	}

	// get regions which are candidates for shadows  
	bool* assumeShadow = new bool[numRegions];
	int* numStrongGrads = new int[numRegions];	
	fill(assumeShadow, assumeShadow + numRegions, 0);
	fill(numStrongGrads, numStrongGrads + numRegions, 0);
	for (int r = 1; r < numRegions; r++)
		for (int l : pSeg->m_borderSegments[r]) {
			// strong gradient in one image that does not occure 
			// in the other image is a hint for a shadow
			if (img1StrongGrad[l] && !img2StrongGrad[l]) {
				numStrongGrads[r]++;
			}
		}

	for (int r = 1; r < numRegions; r++) {
		// check if region contains a sufficient 
		// number of candidate border segments 
		float numBorderSegs = (float)pSeg->m_borderSegments[r].size();
		float percInitiallyDetected = (float)numStrongGrads[r] / numBorderSegs;
		if (percInitiallyDetected > minInitially)
			assumeShadow[r] = 1;
	}

	// for each shadow region candidate get the number of 
	// segments with strong gradients as well as the total 
	// number of segments at the borders to non shadow regions,
	// this is necessary to detect shadows which are split into
	// multiple regions
	int* numSegsAtBorderToShadow = new int[numRegions];
	int* numTrueBorderSegs = new int[numRegions];
	fill(numSegsAtBorderToShadow, numSegsAtBorderToShadow + numRegions, 0);
	fill(numTrueBorderSegs, numTrueBorderSegs + numRegions, 0);
	fill(numStrongGrads, numStrongGrads + numRegions, 0);
	for (int r = 1; r < numRegions; r++) {
		// skip if region is not a candidate for shadows
		//if (!assumeShadow[r]) continue;

		for (int l : pSeg->m_borderSegments[r]) {
			bool atBorderToShadow = false;
			bool strongGrad = false;
			for (int n : neighborhood[l]) {
				// neighbor is assumed to belong to a shadow region
				if (assumeShadow[pSeg->m_mapping.at<int>(n, 0)]) {
					atBorderToShadow = true;
					continue;
				}

				// neighbor is a true border
				numTrueBorderSegs[r]++;
				if (img1StrongGrad[l] && !img2StrongGrad[l])
					strongGrad = true;
			}

			if (atBorderToShadow) numSegsAtBorderToShadow[r]++;
			else if (strongGrad) numStrongGrads[r]++;
		}
	}

	// check if sufficient border segments are detected	
	bool* isShadow = new bool[numRegions];
	fill(isShadow, isShadow + numRegions, 0);
	for (int r = 1; r < numRegions; r++) {
		if (!assumeShadow[r]) continue;
		float numBorderSegs = (float)pSeg->m_borderSegments[r].size();
		float det = (float)(numStrongGrads[r] + numSegsAtBorderToShadow[r]);

		if ((det / numBorderSegs) > minPercentageDetected)
			isShadow[r] = 1;
	}

	// check if region is surrounded by shadows -> it most likely belongs to the shadow
	const float minPercSurroundedByShadow = 0.8;
	for (int r = 1; r < numRegions; r++) {
		float numBorderSegs = (float)pSeg->m_borderSegments[r].size();
		float det = (float)(numStrongGrads[r] + numSegsAtBorderToShadow[r]);
		if ((det / numBorderSegs) > minPercSurroundedByShadow || numSegsAtBorderToShadow[r] == numBorderSegs)
			isShadow[r] = 1;
	}

	ctx.shadows = Mat(labels.size(), CV_8U, Scalar(0));
	ctx.diffEdges = Mat(labels.size(), CV_8U, Scalar(0));
	for (int y = 0; y < labels.rows; y++)
		for (int x = 0; x < labels.cols; x++) {

			if (img1StrongGrad[labels.at<int>(y, x)]
				&& !img2StrongGrad[labels.at<int>(y, x)])
				ctx.diffEdges.at<uchar>(y, x) = 255;

			if (isShadow[pSeg->m_regions.at<int>(y, x)])
				ctx.shadows.at<uchar>(y, x) = 255;
		}

	delete[]	img1StrongGrad, img2StrongGrad, numStrongGrads, 
				numSegsAtBorderToShadow, numTrueBorderSegs, assumeShadow, isShadow;	
}
