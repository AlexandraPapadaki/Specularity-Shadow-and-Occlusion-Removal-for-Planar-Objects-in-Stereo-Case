#include "ImageAlignment.h"


/*	Warp the image from source point to
	target point coordinates
*/
void ImageAlignment::RectifyObject(Mat &img, vector<Point2i> sourcePts, vector<Point2i> targetPts)
{
	// match points to window corners
	vector<Point2i> pts;
	for (Point2i tpt : targetPts) {
		// find closest point
		float minDist = numeric_limits<float>::max();
		int ind = -1, i = 0;
		for (Point2i spt : sourcePts) {
			float dist = sqrt((float)((tpt.x - spt.x)*(tpt.x - spt.x)
											+ (tpt.y - spt.y)*(tpt.y - spt.y)));

			if (dist < minDist) {
				minDist = dist;
				ind = i;
			}
			i++;
		}
		pts.push_back(sourcePts[ind]);
	}
	// warp the object to the entire window
	Mat homography = findHomography(pts, targetPts, 0);
	cv::Mat dst;
	warpPerspective(img, dst, homography, img.size(), InterpolationFlags::INTER_CUBIC);
	img = dst;
}

/*	Aligns source image to reference image by projecting 
	it to the target image plane, the source image is 
	transformed by a homography, which is determined by 
	point correspondences (SIFT feature matches)	
*/
void ImageAlignment::Align(Mat &target, Mat &source)
{
	App::ConsolePrint("Starting Image Alignment");
	
	vector<KeyPoint> tarKps, srcKps;
	Mat tarDesc, srcDesc;
	vector<DMatch> matches;

	// compute feature descriptors
	Ptr<FeatureDetector> pTargetSIFT = xfeatures2d::SIFT::create();
	Ptr<FeatureDetector> pSourceSIFT= xfeatures2d::SIFT::create();
	
	// do this in parallel
	thread thread([pSourceSIFT, &source, &srcKps, &srcDesc]()
	{
		pSourceSIFT->detect(source, srcKps);
		pSourceSIFT->compute(source, srcKps, srcDesc);
	});

	pTargetSIFT->detect(target, tarKps);
	pTargetSIFT->compute(target, tarKps, tarDesc);
	thread.join();

	// match feature descriptors
	Ptr<DescriptorMatcher> pFeatureMatcher = DescriptorMatcher::create("FlannBased");
	pFeatureMatcher->match(srcDesc, tarDesc, matches);
	
	// make point correspondences from matches
	vector<Point2i> tarPts, srcPts;
	for (auto match : matches){
		tarPts.push_back(tarKps[match.trainIdx].pt);
		srcPts.push_back(srcKps[match.queryIdx].pt);
	}

	// apply RANSAC to eliminate outliers, calculate homography 
	// from point correspondences and project the image
	// to the reference image plane, use cubic pixel interpolation
	Mat warped, inliers;
	Mat homography = findHomography(srcPts, tarPts, CV_RANSAC, 2.0, inliers);
	warpPerspective(source, warped, homography, target.size(), CV_INTER_CUBIC);

	// delete information about regions that projected image does not contain
	cv::Mat mask(warped.size(), CV_8U, cv::Scalar(255));
	for (int y = 0; y < warped.rows; y++)
		for (int x = 0; x < warped.cols; x++) {
			if (warped.at<cv::Vec3b>(y, x) == cv::Vec3b::zeros())
				mask.at<uchar>(y, x) = 0;
		}

	m_observable = mask;
	m_warpedImg = warped.clone();
	m_targetImg = ObservableArea(target);

	// count inliers
	int inl = 0;
	for (auto match : matches) {
		if (inliers.at<uchar>(match.queryIdx) == 1) inl++;
	}

	App::ConsolePrint("Image Alignment finished with " + to_string(inl) + " inliers.");
}

