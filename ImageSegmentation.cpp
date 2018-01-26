#include "ImageSegmentation.h"

/*	Apply image segmentation according to the Superpixel Seeds
	Algorithm, the segmentation of img2 is inherited from img1
*/
void ImageSegmentation::Segmentation(
	Mat &img1, Mat &img2, Mat &observable, int numSegments, int iterations)
{
	App::ConsolePrint("Starting image segmentation");

	m_img1 = img1.clone();
	m_img2 = img2.clone();
	m_observable = observable.clone();

	// apply algorithm and get segment labels
	Ptr<ximgproc::SuperpixelSEEDS> pSeeds;
	pSeeds = ximgproc::createSuperpixelSEEDS(m_img1.cols, m_img1.rows, 3, numSegments, 4, 5, 8);
	pSeeds->iterate(m_img1, iterations);

	App::ConsolePrint("SEEDS done");

	Mat labels;
	pSeeds->getLabels(labels);
	int numLabels = pSeeds->getNumberOfSuperpixels();

	// reorder labels
	int numNewLabels = RemapLabels(labels, observable, numLabels);
	m_labels = labels;
	m_numLabels = numNewLabels;
	
	// segment description
	DescribeSegments();
	App::ConsolePrint("Image has been split into " + to_string(m_numLabels) + " segments");
}

/*	make a description for the intermediate result
*/
void ImageSegmentation::DescribeIntermediateRegions(Mat& regions, int numRegions, Mat& regionSizes, Mat& regionValues) {
	// get sizes and BGR intensity values of formed regions
	regionSizes = Mat(numRegions, 1, CV_32S, Scalar(0));
	for (int y = 0; y < regions.rows; y++)
		for (int x = 0; x < regions.cols; x++)
			regionSizes.at<int>(regions.at<int>(y, x), 0)++;

	// 0: BGR 
	// 1: HSV 
	// 2: HLS
	// 3: LAB
	regionValues = Mat(numRegions, 4, CV_32SC3, Scalar(0, 0, 0));

	Mat img1HSV, img1HLS, img1LAB, img2LAB;
	cvtColor(m_img1, img1HSV, CV_BGR2HSV);
	cvtColor(m_img1, img1HLS, CV_BGR2HLS);
	cvtColor(m_img1, img1LAB, CV_BGR2Lab);
	cvtColor(m_img2, img2LAB, CV_BGR2Lab);

	for (int y = 0; y < regions.rows; y++)
		for (int x = 0; x < regions.cols; x++) {
			int r = regions.at<int>(y, x);
			regionValues.at<Vec3i>(r, 0) += (Vec3i)m_img1.at<Vec3b>(y, x);
			regionValues.at<Vec3i>(r, 1) += (Vec3i)img1HSV.at<Vec3b>(y, x);
			regionValues.at<Vec3i>(r, 2) += (Vec3i)img1HLS.at<Vec3b>(y, x);
			regionValues.at<Vec3i>(r, 3) += (Vec3i)img1LAB.at<Vec3b>(y, x);
			Vec3d px = (Vec3d)regionValues.at<Vec3i>(r, 0);
		}

	for (int r = 0; r < numRegions; r++) {
		regionValues.at<Vec3i>(r, 0) /= regionSizes.at<int>(r, 0);
		regionValues.at<Vec3i>(r, 1) /= regionSizes.at<int>(r, 0);
		regionValues.at<Vec3i>(r, 2) /= regionSizes.at<int>(r, 0);
		regionValues.at<Vec3i>(r, 3) /= regionSizes.at<int>(r, 0);
	}
}

/*	Use the segments of the images to form bigger regions by
	merging segments with similar color values and intensities
*/
void ImageSegmentation::MakeRegions(int minSize)
{
	App::ConsolePrint("MakeRegions");

	// make bigger regions by breaking 
	// borders between similar segments
	Mat regions;

	double distTh;
#ifdef MERGING_USE_DIST_BGR
	distTh = MERGE_DIST_BGR;
#endif
#ifdef MERGING_USE_DIST_LAB
		distTh = MERGE_DIST_LAB;
#endif
	int numRegions = MergeToRegions(regions, m_labels, m_numLabels, m_img1Values, distTh);
	Mat regionSizes, regionValues;
	DescribeIntermediateRegions(regions, numRegions, regionSizes, regionValues);
	m_intermediateRegions.push_back(regions.clone());
#ifdef MERGE_TWICE
	numRegions = MergeToRegions(regions, m_labels, m_numLabels, m_img1Values, distTh*MERGE_TWICE_DIST_FACTOR);
	DescribeIntermediateRegions(regions, numRegions, regionSizes, regionValues);
	m_intermediateRegions.push_back(regions.clone());
#endif
#ifdef MERGE_TRIPLE
	numRegions = MergeToRegions(regions, m_labels, m_numLabels, m_img1Values, distTh*MERGE_TRIPLE_DIST_FACTOR);
	DescribeIntermediateRegions(regions, numRegions, regionSizes, regionValues);
	m_intermediateRegions.push_back(regions.clone());
#endif

	// assign all small segments to neighboring segments with sufficient size
	App::ConsolePrint("Assigning segments to regions");

	Mat finalRegions;	
	int maxLabel = AssignSegmentsToRegions(regions, regionValues, minSize, finalRegions);
	int numFinalRegions = RemapLabels(finalRegions, m_observable, maxLabel);

	//describe regions
	Mat img1HSV, img1HLS, img1LAB, img2LAB;
	cvtColor(m_img1, img1HSV, CV_BGR2HSV);
	cvtColor(m_img1, img1HLS, CV_BGR2HLS);
	cvtColor(m_img1, img1LAB, CV_BGR2Lab);
	cvtColor(m_img2, img2LAB, CV_BGR2Lab);

	m_img1RegionValues = Mat(numFinalRegions, 2, CV_32SC3, Scalar(0, 0, 0));
	m_img2RegionValues = Mat(numFinalRegions, 2, CV_32SC3, Scalar(0, 0, 0));
	m_regionDistBGR = Mat(numFinalRegions, 1, CV_32F, Scalar(0));
	m_regionDistLAB = Mat(numFinalRegions, 1, CV_32F, Scalar(0));
	m_regions = finalRegions;
	m_numRegions = numFinalRegions;
	m_regionSizes = Mat(numFinalRegions, 1, CV_32S, Scalar(0));
	for (int y = 0; y < m_regions.rows; y++)
		for (int x = 0; x < m_regions.cols; x++) {
			int r = m_regions.at<int>(y, x);
			m_img1RegionValues.at<Vec3i>(r, 0) += (Vec3i)m_img1.at<Vec3b>(y, x);
			m_img2RegionValues.at<Vec3i>(r, 0) += (Vec3i)m_img2.at<Vec3b>(y, x);
			m_img1RegionValues.at<Vec3i>(r, 1) += (Vec3i)img1LAB.at<Vec3b>(y, x);
			m_img2RegionValues.at<Vec3i>(r, 1) += (Vec3i)img2LAB.at<Vec3b>(y, x);
			m_regionSizes.at<int>(m_regions.at<int>(y, x), 0)++;
		}
	
	for (int r = 0; r < m_numRegions; r++) {
		m_img1RegionValues.at<Vec3i>(r, 0) /= m_regionSizes.at<int>(r, 0);
		m_img2RegionValues.at<Vec3i>(r, 0) /= m_regionSizes.at<int>(r, 0);
		m_img1RegionValues.at<Vec3i>(r, 1) /= m_regionSizes.at<int>(r, 0);
		m_img2RegionValues.at<Vec3i>(r, 1) /= m_regionSizes.at<int>(r, 0);
	}

	// compute intensity from the region values
	Mat m_img1RegionIntensitiesLAB = Mat(numFinalRegions, 1, CV_32F, Scalar(0));
	Mat m_img2RegionIntensitiesLAB = Mat(numFinalRegions, 1, CV_32F, Scalar(0));

	for (int r = 0; r < m_numRegions; r++) {
		Vec3d dBGR = (Vec3d)m_img1RegionValues.at<Vec3i>(r, 0) - (Vec3d)m_img2RegionValues.at<Vec3i>(r, 0);
		Vec3d dLAB = (Vec3d)m_img1RegionValues.at<Vec3i>(r, 1) - (Vec3d)m_img2RegionValues.at<Vec3i>(r, 1);
		m_regionDistBGR.at<float>(r, 0) =
			sqrt(dBGR[0] * dBGR[0] + dBGR[1] * dBGR[1] + dBGR[2] * dBGR[2]);
		m_regionDistLAB.at<float>(r, 0) =
			sqrt(dLAB[0] * dLAB[0] + dLAB[1] * dLAB[1] + dLAB[2] * dLAB[2]);
	}


	// get border segments of each border segment
	MakeBorderSegments();

	// map each segment to the region it belongs to
	m_mapping = Mat(m_numLabels, 1, CV_32S, Scalar(-1));
	for (int y = 0; y < m_regions.rows; y++)
		for (int x = 0; x < m_regions.cols; x++)
			m_mapping.at<int>(m_labels.at<int>(y, x), 0) = m_regions.at<int>(y, x);

	// for each border segment get the neighboring 
	// border segments in the opposite region 
	m_borderNeighborhood = vector<vector<int>>();

	vector<vector<int>> neighborhood;
	GetNeighborhood(m_labels, m_isBorderSegment, neighborhood);

	for (int l = 0; l < neighborhood.size(); l++) {
		vector<int> neighbors = vector<int>();
		for (int n: neighborhood[l]) {
			if (m_mapping.at<int>(l, 0) != m_mapping.at<int>(n, 0))
				neighbors.push_back(n);
		}
		m_borderNeighborhood.push_back(neighbors);
	}
	
	App::ConsolePrint("MakeRegions - done");
}

/*	Rename labels so that they each label 0 to number of
	labels - 1 is used, for further processing we rely on this
	order, non observable part is always mapped to zero label 
*/
int ImageSegmentation::RemapLabels(
	Mat &labels, Mat &observable, int maxLabel)
{
	bool* unusedLabel = new bool[maxLabel + 1];
	int* unobservable = new int[maxLabel + 1];
	int* sizes = new int[maxLabel + 1];

	fill(unusedLabel, unusedLabel + maxLabel + 1, 1);
	fill(unobservable, unobservable + maxLabel + 1, 0);
	fill(sizes, sizes + maxLabel + 1, 0);

	for (int y = 0; y < labels.rows; y++)
		for (int x = 0; x < labels.cols; x++) {
			int l = labels.at<int>(y, x);
			// skip segments that do not belong to observable area
			if (!observable.at<uchar>(y, x)) 
				unobservable[l]++;

			sizes[l]++;
			unusedLabel[l] = 0;
		}

	// map old to new labels
	int* newLabels = new int[maxLabel + 1];
	fill(newLabels, newLabels + maxLabel + 1, -1);
	int numNewLabels = 1;
	for (int l = 0; l < (maxLabel + 1); l++) {
		// map all labels of unobservable to zero label
		if (((float)unobservable[l] / (float)sizes[l]) > 0.5) newLabels[l] = 0;
		else if (!unusedLabel[l]) newLabels[l] = numNewLabels++;		
	}

	// assign new labels
	for (int y = 0; y < labels.rows; y++)
		for (int x = 0; x < labels.cols; x++) 
			labels.at<int>(y, x) = 
				newLabels[labels.at<int>(y, x)];
		
	delete[] unusedLabel, unobservable, sizes, newLabels;

	return numNewLabels;
}

/*	Make a description for each segment, a segment is described
	by an average color value and it's intensity, the size of
	the segment in pixels as well as the difference in it's 
	intensity value to the corresponding segment in the other image
*/
void ImageSegmentation::DescribeSegments()
{
	// 0: BGR 
	// 1: HSV 
	// 2: HLS
	// 3: LAB
	m_img1Values = Mat(m_numLabels, 4, CV_32SC3, Scalar(0, 0, 0));
	m_img2Values = Mat(m_numLabels, 4, CV_32SC3, Scalar(0, 0, 0));
	m_img1ValuesGRAY = Mat(m_numLabels, 1, CV_32SC1, Scalar(0));
	m_img2ValuesGRAY = Mat(m_numLabels, 1, CV_32SC1, Scalar(0));

	m_img1IntensitiesBGR = Mat(m_numLabels, 1, CV_32FC1, Scalar(0.0));
	m_img2IntensitiesBGR = Mat(m_numLabels, 1, CV_32FC1, Scalar(0.0));
	m_intensitiesDiffBGR = Mat(m_numLabels, 1, CV_32FC1, Scalar(0.0));
	m_segSizes = Mat(m_numLabels, 1, CV_32SC1, Scalar(0));

	Mat img1HSV, img2HSV, img1HLS, img2HLS, img1LAB, img2LAB, img1GRAY, img2GRAY;
	cvtColor(m_img1, img1HSV, CV_BGR2HSV);
	cvtColor(m_img2, img2HSV, CV_BGR2HSV);
	cvtColor(m_img1, img1HLS, CV_BGR2HLS);
	cvtColor(m_img2, img2HLS, CV_BGR2HLS);
	cvtColor(m_img1, img1LAB, CV_BGR2Lab);
	cvtColor(m_img2, img2LAB, CV_BGR2Lab);
	cvtColor(m_img1, img1GRAY, CV_BGR2GRAY);
	cvtColor(m_img2, img2GRAY, CV_BGR2GRAY);

	// sum up each pixel value of a segment
	for (int y = 0; y < m_labels.rows; y++)
		for (int x = 0; x < m_labels.cols; x++) {
			int label = m_labels.at<int>(y, x);	
			m_img1Values.at<Vec3i>(label, 0) += m_img1.at<Vec3b>(y, x);
			m_img1Values.at<Vec3i>(label, 1) += img1HSV.at<Vec3b>(y, x);
			m_img1Values.at<Vec3i>(label, 2) += img1HLS.at<Vec3b>(y, x);
			m_img1Values.at<Vec3i>(label, 3) += img1LAB.at<Vec3b>(y, x);
			m_img1ValuesGRAY.at<int>(label, 0) += (int)img1GRAY.at<uchar>(y, x);
			m_img2Values.at<Vec3i>(label, 0) += m_img2.at<Vec3b>(y, x);
			m_img2Values.at<Vec3i>(label, 1) += img2HSV.at<Vec3b>(y, x);
			m_img2Values.at<Vec3i>(label, 2) += img2HLS.at<Vec3b>(y, x);
			m_img2Values.at<Vec3i>(label, 3) += img2LAB.at<Vec3b>(y, x);
			m_img2ValuesGRAY.at<int>(label, 0) += (int)img2GRAY.at<uchar>(y, x);
			m_segSizes.at<int>(label, 0)++;
		}

	//	averaging
	for (int l = 0; l < m_numLabels; l++) {
		m_img1Values.at<Vec3i>(l, 0) /= m_segSizes.at<int>(l, 0);
		m_img1Values.at<Vec3i>(l, 1) /= m_segSizes.at<int>(l, 0);
		m_img1Values.at<Vec3i>(l, 2) /= m_segSizes.at<int>(l, 0);
		m_img1Values.at<Vec3i>(l, 3) /= m_segSizes.at<int>(l, 0);
		m_img1ValuesGRAY.at<int>(l, 0) /= m_segSizes.at<int>(l, 0);
		m_img2Values.at<Vec3i>(l, 0) /= m_segSizes.at<int>(l, 0);
		m_img2Values.at<Vec3i>(l, 1) /= m_segSizes.at<int>(l, 0);
		m_img2Values.at<Vec3i>(l, 2) /= m_segSizes.at<int>(l, 0);
		m_img2Values.at<Vec3i>(l, 3) /= m_segSizes.at<int>(l, 0);
		m_img2ValuesGRAY.at<int>(l, 0) /= m_segSizes.at<int>(l, 0);
	}

	// compute intensity from the segment values
	for (int l = 0; l < m_numLabels; l++) {
		Vec3i vals1 = m_img1Values.at<Vec3i>(l, 0);
		Vec3i vals2 = m_img2Values.at<Vec3i>(l, 0);

		m_img1IntensitiesBGR.at<float>(l, 0) =
			sqrt(vals1[0] * vals1[0] + vals1[1] * vals1[1] + vals1[2] * vals1[2]);

		m_img2IntensitiesBGR.at<float>(l, 0) =
			sqrt(vals2[0] * vals2[0] + vals2[1] * vals2[1] + vals2[2] * vals2[2]);
	}
	m_intensitiesDiffBGR = m_img1IntensitiesBGR - m_img2IntensitiesBGR;
}

/*	Form connected regions of similar segments by merging adjacent
	segments with a color vector difference within threshold borderTh 
	and intensity difference within threshold distBGR 
*/
int ImageSegmentation::MergeToRegions(Mat& regions, Mat& labels, int numLabels, Mat& imgValues, double distTh)
{
	App::ConsolePrint("MergeToRegions border threshold: " + to_string(distTh));
	// for each segment find its neighbors
	vector<vector<int>> segmentNeighborhood;
	Mat segmentsOfInterest(numLabels, 1, CV_8U, Scalar(1));
	GetNeighborhood(labels, segmentsOfInterest, segmentNeighborhood);
	
	// determine similar neighboring segments to merge
	vector<vector<int>> mergeWith;
	for (int l = 0; l < numLabels; l++)
		mergeWith.push_back(vector<int>());

	bool* hasMaster = new bool[numLabels];
	fill(hasMaster, hasMaster + numLabels, 0);

	for (int l = 0; l < numLabels; l++) {
		// skip if segment will be merged into another segment already
		if (hasMaster[l]) continue;

		for (int neighbor : segmentNeighborhood[l]) {
			// merge only if segments are similar
			double dist;
#ifdef MERGING_USE_INTENSITY_DIFF_BGR
			Vec3i dVal = imgValues.at<Vec3i>(l, 0) - imgValues.at<Vec3i>(neighbor, 0);
			if (abs(dVal[0]) > MERGE_BORDER_TH[0] || abs(dVal[1]) > MERGE_BORDER_TH[1] || abs(dVal[2]) > MERGE_BORDER_TH[2])
				continue;
			dist = abs(m_img1IntensitiesBGR.at<float>(l, 0) - m_img1IntensitiesBGR.at<float>(neighbor, 0));
#endif
#ifdef MERGING_USE_DIST_BGR
			Vec3d dBGR = (Vec3d)imgValues.at<Vec3i>(l, 0) - (Vec3d)imgValues.at<Vec3i>(neighbor, 0);
			dist = sqrt(dBGR[0] * dBGR[0] + dBGR[1] * dBGR[1] + dBGR[2] * dBGR[2]);
#endif
#ifdef MERGING_USE_DIST_LAB
			Vec3d dLAB = (Vec3d)imgValues.at<Vec3i>(l, 3) - (Vec3d)imgValues.at<Vec3i>(neighbor, 3);
			dist = sqrt(dLAB[0] * dLAB[0] + dLAB[1] * dLAB[1] + dLAB[2] * dLAB[2]);
#endif
			if (dist > distTh) continue;

			// neighbor has a master -> merge the entire region to l
			if (hasMaster[neighbor]) {
				// reference to master segment
				int master = mergeWith[neighbor][0];
				// skip if l is already the master of this neighbor 
				if (master == l) continue;
				// set new master
				mergeWith[l].push_back(master);
				for (int slave : mergeWith[master]) {
					mergeWith[l].push_back(slave);
					mergeWith[slave][0] = l;
				}
				// old master is slave of new master
				mergeWith[master].clear();
				mergeWith[master].push_back(l);
				hasMaster[master] = 1;
			}
			// neighbor is a master
			else {
				mergeWith[l].push_back(neighbor);
				// if neighbor has slaves merge the entire region to l
				if (!mergeWith[neighbor].empty()) {
					for (int slave : mergeWith[neighbor]) {
						mergeWith[l].push_back(slave);
						mergeWith[slave][0] = l;
					}
				}
				// set master
				mergeWith[neighbor].clear();
				mergeWith[neighbor].push_back(l);
				hasMaster[neighbor] = 1;
			}
			hasMaster[l] = 0;
		}
	}

	// assign new labels to masters 
	int newLabel = 0;
	int* newLabels = new int[numLabels];
	for (int l = 0; l < numLabels; l++) {
		if (hasMaster[l])
			newLabels[l] = -1;
		else
			newLabels[l] = newLabel++;
	}

	// assign master's labels to their slaves
	for (int l = 0; l < numLabels; l++) {
		if (newLabels[l] == -1)
			newLabels[l] = newLabels[mergeWith[l][0]];
	}

	// merge segment labels
	regions = labels.clone();
	for (auto it = regions.begin<int>(); it != regions.end<int>(); it++) *it = newLabels[*it];

	delete[] hasMaster, newLabels;

	App::ConsolePrint("MergeToRegions done");
	return newLabel;
}


/*	After MergeToRegions big regions have been formed from segments with
	similar values and small segments with high intensity differences remain,
	therefore assign the small segments to the most similar neighboring region
*/
int ImageSegmentation::AssignSegmentsToRegions(
	Mat& regions, Mat& regionValues, int sizeTh, Mat& finalRegions)
{
	int numLabels = regionValues.rows;
	finalRegions = regions.clone();

	// find the sizes of the intermediate regions
	Mat sizes(numLabels, 1, CV_32S, Scalar(0));
	for (int y = 0; y < regions.rows; y++)
		for (int x = 0; x < regions.cols; x++)
			sizes.at<int>(regions.at<int>(y, x), 0)++;

	// find the small segments that have to be assigned
	Mat smallSegments(numLabels, 1, CV_8U, Scalar(0));
	int remaining = 0;
	for (int l = 1; l < numLabels; l++) {
		if (sizes.at<int>(l, 0) < sizeTh){
			smallSegments.at<uchar>(l, 0) = 1;
			remaining++;
		}
	}

	int forceExit = 10;

	// iteratively assign small segments to master  
	// segments until no small segments are left
	int* mergeTo = new int[numLabels];
	bool* mergeThisIteration = new bool[numLabels];
	fill(mergeTo, mergeTo + numLabels, -1);
	while (remaining != 0) { // ?
		if (forceExit-- < 0) {
			cout << "WARNING: AssignSegmentsToRegions - Exiting with " << remaining << " segments remaining!" << endl;
			break;
		}
		vector<vector<int>> neighborhood;
		GetNeighborhood(finalRegions, smallSegments, neighborhood);	

		// find neighboring master segment with most similar color value
		fill(mergeThisIteration, mergeThisIteration + numLabels, 0);
		for (int l = 1; l < numLabels; l++) {
			vector<int> neighbors = neighborhood[l];
			// skip if not a small segment
			if (neighbors.empty() || mergeThisIteration[l]) continue;

			int L = -1;
			float lowestDist = numeric_limits<float>::max();
			for (int n : neighbors) {
				// only merge to a big segment
				if (smallSegments.at<uchar>(n, 0)) continue;
				//if (mergeThisIteration[n]) continue;

				double dist;
#ifdef MERGING_USE_INTENSITY_DIFF_BGR
				dist = abs(regionIntensitiesBGR.at<float>(l, 0) - regionIntensitiesBGR.at<float>(n, 0));
#endif
#ifdef MERGING_USE_DIST_BGR
				Vec3d dBGR = (Vec3d)regionValues.at<Vec3i>(l, 0) - (Vec3d)regionValues.at<Vec3i>(n, 0);
				dist = sqrt(dBGR[0] * dBGR[0] + dBGR[1] * dBGR[1] + dBGR[2] * dBGR[2]);
#endif
#ifdef MERGING_USE_DIST_LAB
				Vec3d dLAB = (Vec3d)regionValues.at<Vec3i>(l, 3) - (Vec3d)regionValues.at<Vec3i>(n, 3);
				dist = sqrt(dLAB[0] * dLAB[0] + dLAB[1] * dLAB[1] + dLAB[2] * dLAB[2]);
#endif

				if (dist < lowestDist) {
					lowestDist = dist;
					L = n;
				}
			}
			mergeTo[l] = L;
			if (L == -1) continue;

			// debug
			//if (mergeThisIteration[L]) {
			//	cout << "something went wrong" << "with master" << endl;
			//}
			//if (mergeThisIteration[l]) {
			//	cout << "something went wrong" << "with slave" << endl;
			//}
			//mergeThisIteration[l] = 1;
			//mergeThisIteration[L] = 1;
		}

		// rename the labels of the segments that have to be merged
		for (int y = 0; y < finalRegions.rows; y++)
			for (int x = 0; x < finalRegions.cols; x++) {
				int l = finalRegions.at<int>(y, x);
				if (mergeTo[l] != -1) {
					finalRegions.at<int>(y, x) = mergeTo[l];
					if(smallSegments.at<uchar>(l, 0)){
						remaining--;
						smallSegments.at<uchar>(l, 0) = 0;
					}
					// set new values and sizes
					int sizeL = sizes.at<int>(l, 0),
						sizeR = sizes.at<int>(mergeTo[l], 0);
					int newSize = sizeL + sizeR;
					sizes.at<int>(l, 0) = newSize;
					sizes.at<int>(mergeTo[l], 0) = newSize;
					double relSizeL = (double)sizeL / (double)newSize;
#ifdef MERGING_USE_INTENSITY_DIFF_BGR	
					float newVal = relSizeL * regionIntensitiesBGR.at<float>(l, 0)
						+ (1.0 - relSizeL) * regionIntensitiesBGR.at<float>(mergeTo[l], 0);
					regionIntensitiesBGR.at<float>(l, 0) = newVal;
					regionIntensitiesBGR.at<float>(mergeTo[l], 0) = newVal;

#endif					
#ifdef MERGING_USE_DIST_BGR	
					Vec3i newVal = relSizeL * (Vec3d)regionValues.at<Vec3i>(l, 0)
						+ (1.0 - relSizeL) * (Vec3d)regionValues.at<Vec3i>(mergeTo[l], 0);
					regionValues.at<Vec3i>(l, 0) = newVal;
					regionValues.at<Vec3i>(mergeTo[l], 0) = newVal;
#endif
#ifdef MERGING_USE_DIST_LAB
					Vec3i newVal = relSizeL * (Vec3d)regionValues.at<Vec3i>(l, 3)
						+ (1.0 - relSizeL) * (Vec3d)regionValues.at<Vec3i>(mergeTo[l], 3);
					regionValues.at<Vec3i>(l, 3) = newVal;
					regionValues.at<Vec3i>(mergeTo[l], 3) = newVal;
#endif
				}
			}
	}

	delete[] mergeTo;

	// get max label
	double unused, maxLabel;
	minMaxIdx(finalRegions, &unused, &maxLabel);
	return (int)maxLabel;
}

/*	Returns the adjacent segments for each segment specified in 
	segmentsOfInterest, neighborhood: outer vector over all segment labels,
	inner vector contains labels of adjacent segments (is empty when not 
	specified in segmentsOfInterest)
*/
void ImageSegmentation::GetNeighborhood(
	Mat &labels, Mat &segmentsOfInterest, vector<vector<int>> &neighborhood)
{
	int numLabels = segmentsOfInterest.rows;

	// for each segment find its neighbors
	vector<vector<int>> segmentNeighborhood;
	for (int l = 0; l < numLabels; l++) 
		segmentNeighborhood.push_back(vector<int>());

	// divide image into slices 
	int sliceSize = labels.rows / NUM_THREADS;
	int sliceStart = 0;

	vector<thread*> threads;
	mutex mtx;	
	// scan each slice in an own thread, threading here
	// provides huge speedup in case of larger images
	for (int n = 0; n < NUM_THREADS; n++) {
		if (n == NUM_THREADS - 1)
			// last slice 
			sliceSize = labels.rows - sliceStart - 1;

		shared_ptr<Mat> pSlice = make_shared<Mat>(
			labels(Rect(0, sliceStart, labels.cols, sliceSize)).clone());
		shared_ptr<Mat> pSegmentsOfInterest = make_shared<Mat>(segmentsOfInterest);

		thread* pTh = new thread([pSlice, pSegmentsOfInterest, &segmentNeighborhood, &mtx]()
		{
			for (int y = 1; y < (pSlice->rows - 1); y++)
				for (int x = 1; x < (pSlice->cols - 1); x++) {
					vector<int> adjacent;
					int l0 = pSlice->at<int>(y, x), l1;

					// skip if neighborhood is not desired for that segment
					if (!pSegmentsOfInterest->at<uchar>(l0, 0)) continue;

					// check for local label borders
					if (l0 != (l1 = pSlice->at<int>(y, x + 1)) && l1 != 0)
						adjacent.push_back(l1);
					if (l0 != (l1 = pSlice->at<int>(y, x - 1)) && l1 != 0)
						adjacent.push_back(l1);
					if (l0 != (l1 = pSlice->at<int>(y + 1, x)) && l1 != 0)
						adjacent.push_back(l1);
					if (l0 != (l1 = pSlice->at<int>(y - 1, x)) && l1 != 0)
						adjacent.push_back(l1);

					for (int adj : adjacent) {
						// only store neighbor once
						bool exists = 0;	
						mtx.lock();
						for (int l : segmentNeighborhood[l0]) {
							if (l == adj) {
								exists = true;
								break;
							}
						}				
						if (!exists) segmentNeighborhood[l0].push_back(adj);
						mtx.unlock();
					}
				}
		});

		threads.push_back(pTh);
		sliceStart += sliceSize;
	}

	// synchronize
	for (int n = 0; n < NUM_THREADS; n++) 
		threads[n]->join(); 

	neighborhood = segmentNeighborhood;
}

/*	Keep track of all the segments which
	make up the border of the regions
*/
void ImageSegmentation::MakeBorderSegments()
{
	m_borderSegments.clear();
	m_isBorderSegment = Mat(m_numLabels, 1, CV_8U, Scalar(0));
	for (int r = 0; r < m_numRegions; r++) {
		m_borderSegments.push_back(vector<int>());
	}
	
	// mark segments as border if they belong to region border
	for (int y = 1; y < (m_regions.rows - 1); y++)
		for (int x = 1; x < (m_regions.cols - 1); x++) {	
			int r = m_regions.at<int>(y, x);
			int l = m_labels.at<int>(y, x);

			if (m_regions.at<int>(y + 1, x) != r && m_regions.at<int>(y + 1, x) != 0 ||
				m_regions.at<int>(y - 1, x) != r && m_regions.at<int>(y - 1, x) != 0 ||
				m_regions.at<int>(y, x + 1) != r && m_regions.at<int>(y, x + 1) != 0 ||
				m_regions.at<int>(y, x - 1) != r && m_regions.at<int>(y, x - 1) != 0) 

					m_isBorderSegment.at<uchar>(l, 0) = 1;							
		}

	// map each segment to the region it belongs to
	bool* mapped = new bool[m_numLabels];
	fill(mapped, mapped + m_numLabels, 0);
	for (int y = 0; y < m_labels.rows; y++)
		for (int x = 0; x < m_labels.cols; x++) {
			int l = m_labels.at<int>(y, x);
			// only store label once
			if(mapped[l]) continue;
			// skip if it is not a border segment
			if (!m_isBorderSegment.at<uchar>(l, 0)) continue;

			int r = m_regions.at<int>(y, x);
			m_borderSegments[r].push_back(l);
			mapped[l] = true;
		}

	delete[] mapped;
}
