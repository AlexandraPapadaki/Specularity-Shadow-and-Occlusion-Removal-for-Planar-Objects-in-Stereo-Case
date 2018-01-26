#include "SpecularityRemoval.h"

/*	Constructor that initializes the class with all
	necessary information in order to allow thread
	safe execution after initialization
*/
SpecularityRemoval::SpecularityRemoval(
	SpecularityDetection* pDet)
{
	m_pDet = pDet;
	m_tarImg = pDet->m_tarImg.clone();
	m_srcImg = pDet->m_srcImg.clone();
	m_areaToCorrect = pDet->MakeCorrectionRegion();
}

/*	Copy information that need to be corrected from 
	the source to the target image, in order to smooth  
	the borders apply Poisson Blending to blend the 
	border region back into the target image
*/
Mat SpecularityRemoval::MakeCorrectedImage() {
	return MakeCorrectedImage(Mat(), Mat());
}

Mat SpecularityRemoval::MakeCorrectedImage(Mat& blendingBorder, Mat& correctionArea)
{
	// exclude small areas which might occur due 
	// to alignment issues, or misdetections
	RemoveSmallComponents();

	// TODO
	Size expKernelSize(3, 3);
	int expIterations = 10;
	Mat blendingMask = MakeBorderBlendingMask(expKernelSize, expIterations);
	//Mat blendingMask = MakeBorderBlendingMask();

	// get the center of blending area
	Point2i ct(0, 0);
	int left = blendingMask.cols, right = 0,
		upper = blendingMask.rows, lower = 0;
	bool isEmpty = true;
	for (int y = 0; y < blendingMask.rows; y++)
		for (int x = 0; x < blendingMask.cols; x++) {
			if (!blendingMask.at<uchar>(y, x)) continue;
			isEmpty = false;
			if (left > x) left = x;
			if (right < x) right = x;
			if (upper > y) upper = y;
			if (lower < y) lower = y;
		}
	ct.x = (left + right) / 2;
	ct.y = (upper + lower) / 2;
	
	if (isEmpty) {
		cout << "Nothing to correct. Exiting!" << endl;
		return Mat();
	}

	// copy source image area to target image and blend borders
	Mat corrected = m_tarImg.clone();
	m_srcImg.copyTo(corrected, m_areaToCorrect);
	App::DisplayImage(corrected, "corrected", -1);
	seamlessClone(m_srcImg, corrected, blendingMask, ct, corrected, NORMAL_CLONE);

	blendingBorder = Mat(m_tarImg.size(), m_tarImg.type());
	correctionArea = Mat(m_tarImg.size(), m_tarImg.type());
	m_tarImg.copyTo(blendingBorder, blendingMask);
	m_tarImg.copyTo(correctionArea, m_areaToCorrect);

	return corrected;
}

/*	Detect small blobs and holes (pixelsize smaller than N) 
	in the correction mask and fill them
	
*/
void SpecularityRemoval::RemoveSmallComponents()
{
	const int nb = (m_tarImg.cols * m_tarImg.rows) * MAX_BLOB_SIZE_FRACTION;
	const int nh = (m_tarImg.cols * m_tarImg.rows) * MAX_HOLE_SIZE_FRACTION;

	Mat area = m_areaToCorrect.clone();
	App::DisplayImage(m_areaToCorrect, "m_areaToCorrect before", -1);
	// since pixels that are not included in the mask are
	// labeled zero we need to detect the components for
	// the inverted mask to label holes
	Mat cLabels, ncLabels;
	int cNum = connectedComponents(area, cLabels);
	int ncNum = connectedComponents(255 - area, ncLabels);

	// figure out size of each component
	int* cSizes = new int[cNum];
	int* ncSizes = new int[ncNum];
	fill(cSizes, cSizes + cNum, 0);
	fill(ncSizes, ncSizes + ncNum, 0);

	for (int y = 0; y < area.rows; y++)
		for (int x = 0; x < area.cols; x++) {
			cSizes[cLabels.at<int>(y, x)]++;
			ncSizes[ncLabels.at<int>(y, x)]++;
		}

	// a blob surrounded by a hole and a hole
	// surrounded by a blob needs to be skipped
	bool* cSkip = new bool[cNum];
	bool* ncSkip = new bool[ncNum];
	fill(cSkip, cSkip + cNum, 0);
	fill(ncSkip, ncSkip + ncNum, 0);

	for (int y = 0; y < area.rows; y++)
		for (int x = 1; x < area.cols; x++) {
			if (cSizes[cLabels.at<int>(y, x)] < nb
				// checking for a random pixel at the border is sufficient
				&& ncSizes[ncLabels.at<int>(y, x - 1)] < nh) {
				cSkip[cLabels.at<int>(y, x)] = 1;
			}
			if (ncSizes[ncLabels.at<int>(y, x)] < nh
				&& cSizes[cLabels.at<int>(y, x - 1)] < nb)
				ncSkip[ncLabels.at<int>(y, x)] = 1;
		}

	for (int y = 0; y < area.rows; y++)
		for (int x = 0; x < area.cols; x++) {
			int l1 = cLabels.at<int>(y, x);
			int l2 = ncLabels.at<int>(y, x);

			// point belongs to blob 
			if (cSizes[l1] < nb && !cSkip[l1])
				area.at<uchar>(y, x) = 0;
			// point belongs to hole 
			if (ncSizes[l2] < nh && !ncSkip[l2])
				area.at<uchar>(y, x) = 255;
		}

	delete[] cSizes, ncSizes, cSkip, ncSkip;
	m_areaToCorrect = area;
	App::DisplayImage(m_areaToCorrect, "m_areaToCorrect after", 100);
}

Mat SpecularityRemoval::MakeBorderBlendingMask()
{
	
	// slightly dilate correction area
	// TODO:
	Mat expansionKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	const int expIterations = 10;
	dilate(m_areaToCorrect, m_areaToCorrect, expansionKernel, Point(-1, -1), expIterations);

	ImageSegmentation* pSeg = m_pDet->m_pTarSrcSegmentation;
	
	// find segments that lie in expanded correction area
	Mat labels = pSeg->m_labels;
	Mat segmentsOfInterest(pSeg->m_numLabels, 1, CV_8U, Scalar(0));
	for (int y = 0; y < m_areaToCorrect.rows; y++)
		for (int x = 0; x < m_areaToCorrect.cols; x++) {
			int l = labels.at<int>(y, x);
			if (m_areaToCorrect.at<uchar>(y, x)) {
				segmentsOfInterest.at<uchar>(l, 0) = 1;
			}
		}

	// TODO:
	const int BLENDING_BORDER_THICKNESS = 2;
	Mat expandedAreaToCorrect = m_areaToCorrect.clone();
	for (int i = 0; i < BLENDING_BORDER_THICKNESS; i++) {
		vector<vector<int>> segmentNeighborhood;
		pSeg->GetNeighborhood(labels, segmentsOfInterest, segmentNeighborhood);

		for(auto neighbors: segmentNeighborhood)
			for (int n : neighbors) 
				segmentsOfInterest.at<uchar>(n, 0) = 1;

		if (i == 0) {
			for (int y = 0; y < expandedAreaToCorrect.rows; y++)
				for (int x = 0; x < expandedAreaToCorrect.cols; x++) {
					int l = labels.at<int>(y, x);
					if (segmentsOfInterest.at<uchar>(l, 0))
						expandedAreaToCorrect.at<uchar>(y, x) = 255;
				}
		}
	}

	Mat borderBlendingMask(m_areaToCorrect.size(), CV_8U, Scalar(0));
	for (int y = 0; y < borderBlendingMask.rows; y++)
		for (int x = 0; x < borderBlendingMask.cols; x++) {
			int l = labels.at<int>(y, x);
			if (segmentsOfInterest.at<uchar>(l, 0)) 
				borderBlendingMask.at<uchar>(y, x) = 255;
		}

	borderBlendingMask &= (255 - m_areaToCorrect);
	m_areaToCorrect = expandedAreaToCorrect;
	return borderBlendingMask;
}


/*	Produce a border region mask by shrinking and
	expanding the correction region with dilation 
	and erosion operations
*/
Mat SpecularityRemoval::MakeBorderBlendingMask(
	Size kernelSize, int k)
{
	Mat expansionKernel = getStructuringElement(MORPH_ELLIPSE, kernelSize);

	dilate(m_areaToCorrect, m_areaToCorrect, expansionKernel, Point(-1, -1), 2 * k);

	Mat shrink, expand, borderMask;
	erode(m_areaToCorrect, shrink, expansionKernel, Point(-1, -1), k);
	dilate(m_areaToCorrect, expand, expansionKernel, Point(-1, -1), k);

	Mat refine = getStructuringElement(MORPH_RECT, Size(2, 2));
	borderMask = expand & abs(shrink - 255);
	dilate(borderMask, borderMask, refine, Point(-1, -1), 10);

	return borderMask;
}