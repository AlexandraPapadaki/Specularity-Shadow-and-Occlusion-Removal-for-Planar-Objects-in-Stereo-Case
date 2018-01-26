#include "App.h"
#include "ImageAlignment.h"
#include "SpecularityDetection.h"
#include "SpecularityRemoval.h"


// some static references, because we 
// dont want any instances
static App app;
static ImageAlignment imgAlign;
static SpecularityDetection* pSpecDetection = new SpecularityDetection();

App::SegmentComparison App::getScSpec() {
	return app.scSpec;
}
App::SegmentComparison App::getScShadows() {
	return app.scShadows;
}

/*	Initialize context of the application, 
	read the images and rescale them if required
*/
void App::InitApp(Mat &tar, Mat &src) 
{
	tar = imread(TARGET_IMAGE_PATH);
	src = imread(SOURCE_IMAGE_PATH);

	int width, height;
	if ((width = tar.cols) != src.cols || (height = tar.rows) != src.rows) {
		cerr << "Image dimensions do not match. Exiting!" << endl;
		exit(-1); 
	}

	// resize images if too large
	double ratioX = (double)MAX_IMAGE_SIZE / (double)width;
	double ratioY = (double)MAX_IMAGE_SIZE / (double)height;
	double scale = min(ratioX, ratioY);

	if (scale < 1.0) {
		resize(tar, tar, Size(width * scale, height * scale));
		resize(src, src, Size(width * scale, height * scale));
	}

	// get scale for displaying images
	width = tar.cols;
	height = tar.rows;
	app.displayingScale = 1.0;
	if (max(width, height) > MAX_DISPLAYING_SIZE) {
		ratioX = (double)MAX_DISPLAYING_SIZE / (double)width;
		ratioY = (double)MAX_DISPLAYING_SIZE / (double)height;
		app.displayingScale = min(ratioX, ratioY);
	}

	app.tarImg = tar;
	app.srcImg = src;
}

/*	Print in mutable console
*/
void App::ConsolePrint(std::string msg)
{
	app.console_mtx->lock();
	cout << msg << endl;
	app.console_mtx->unlock();
}

/*	Display images in reasonable size
*/
void App::DisplayImage(Mat &img, std::string win, int ms)
{
	if (img.empty()) {
		cout << "Displayed image is empty" << endl;
		return;
	}

	Mat show;
	if(app.displayingScale != 1.0)
		resize(img, show, Size(	img.cols * app.displayingScale,
										img.rows * app.displayingScale));
	else show = img;
	namedWindow(win);
	imshow(win, show);
	if(ms != -1) waitKey(ms);
}


/*	Start the rectification process
	if desired and align images 
*/
void App::RunAlignment(bool rectification)
{
	// rectify
	if (rectification) {
		int h = app.tarImg.rows, w = app.tarImg.cols;

		// window borders
		vector<Point2i> winPts;
		Point2i ul(0, 0), ur(w - 1, 0), ll(0, h - 1), lr(w - 1, h - 1);
		winPts.push_back(ul);
		winPts.push_back(ur);
		winPts.push_back(ll);
		winPts.push_back(lr);

		destroyAllWindows();
		// wait for user to click 4 points
		vector<Point2i> trackedPts;
		// target image
		trackedPts = App::StartTracking(app.tarImg);
		imgAlign.RectifyObject(app.tarImg, trackedPts, winPts);
		App::DisplayImage(app.tarImg, "Rectify Image", 0);
		// source image
		trackedPts = App::StartTracking(app.srcImg);
		imgAlign.RectifyObject(app.srcImg, trackedPts, winPts);
		App::DisplayImage(app.srcImg, "Rectify Image", 0);
	}

	// align 
	imgAlign.Align(app.tarImg, app.srcImg);
	app.tarImg = imgAlign.m_targetImg;
	app.srcImg = imgAlign.m_warpedImg;
	DisplayImage(app.tarImg, "target image", -1);
	DisplayImage(app.srcImg, "source image", 500);
}

/*	Start segmentation, initialize detection
*/
void App::SetupAlgorithm(bool tarHasShadows, bool srcHasShadows)
{
	// initialize shadow detection first
	pSpecDetection->Initialize(app.tarImg, app.srcImg, 
		imgAlign.m_observable, tarHasShadows, srcHasShadows);
}

/*	Wait for user interaction,
	accept result with ENTER
*/
bool waitForAction(bool &changed) {
	int button = 0;
	while (1) {
		button = waitKey(100);
		if (changed) {
			changed = false;
			return 1;
		}
		if (button == BUTTON_OK) 
			return 0;
	}
}

/*	Simple callback
*/
static void onShadowDetectionParamsChanged(int, void* args) {
	bool* changed = (bool*)args;
	*changed = true;
}

/*	Initialze a small GUI and a shadow detection 
	context, run detection in the desired images
*/
void App::RunShadowDetection()
{
	if (!pSpecDetection->m_bTarHasShadows && !pSpecDetection->m_bSrcHasShadows) return;
	cout << endl << "Running shadow detection" << endl;
	//app.scShadows = App::SelectSegmentComparison();
	//cout << "Selected method: " << App::scToString(app.scShadows) << endl;

	const std::string win1 = "Differing Edge Segments";
	const std::string win2 = "Shadow Detection";
	
	bool changed = false;
	// create simple GUI
	namedWindow(win2);
	createTrackbar("Grad Thresh", win2, &app.sdc.gradTh, 50, &onShadowDetectionParamsChanged, &changed);
	createTrackbar("Minimum Initial %", win2, &app.sdc.minInitiallyDetected, 50, &onShadowDetectionParamsChanged, &changed);
	createTrackbar("Borders Detected %", win2, &app.sdc.minPercentageDetected, 100, &onShadowDetectionParamsChanged, &changed);

	// run shadow detection in target image
	if (pSpecDetection->m_bTarHasShadows){
		app.sdc.ResetParams();
		ImageSegmentation* pSeg = pSpecDetection->m_pTarSrcSegmentation;
		// display target region segmentation
		Mat borders = UTIL::drawLabelBorders(pSeg->m_regions, app.tarImg, COLOR_BLACK);
		App::DisplayImage(borders, "Target Image Segmentation", -1);

		// INTERMEDIATE RESULT
		cv::imwrite("regions_tar.png", borders);
		borders = UTIL::drawLabelBorders(pSeg->m_labels, app.tarImg, COLOR_BLACK);
		cv::imwrite("segments_tar.png", borders);
		for (int i = 0; i < pSeg->m_intermediateRegions.size(); i++) {
			borders = UTIL::drawLabelBorders(pSeg->m_intermediateRegions[i], app.tarImg, COLOR_BLACK);
			cv::imwrite("intermediate_regions_tar_" + to_string(i) + ".png", borders);
		}
		// run detection with current parameters
		Mat shadows;
		while (1) {
			pSpecDetection->DetectTargetShadows(app.sdc);
			Mat displayEdges = Mat(app.tarImg.size(), app.srcImg.type(), COLOR_WHITE);
			app.tarImg.copyTo(displayEdges, app.sdc.diffEdges);
			//displayEdges = UTIL::drawLabelBorders(pSeg->m_labels, displayEdges, COLOR_BLUE);
			displayEdges = UTIL::drawLabelBorders(pSeg->m_regions, displayEdges, COLOR_BLACK, 3);
			DisplayImage(displayEdges, win1, -1);

			shadows = Mat(app.tarImg.size(), CV_8UC3, Scalar(255, 255, 255));
			app.tarImg.copyTo(shadows, app.sdc.shadows);
			DisplayImage(shadows, win2, -1);
			if (!waitForAction(changed)) break;
		}
		// INTERMEDIATE RESULT
		cv::imwrite("shadow_tar.png", shadows);

		pSpecDetection->m_tarShadows = app.sdc.shadows;
		destroyWindow("Target Image Segmentation");
	}

	// run shadow detection in source image
	if (pSpecDetection->m_bSrcHasShadows){
		app.sdc.ResetParams();
		ImageSegmentation* pSeg = pSpecDetection->m_pSrcTarSegmentation;
		// display source region segmentation
		Mat borders = UTIL::drawLabelBorders(pSeg->m_regions, app.srcImg, COLOR_BLACK);
		App::DisplayImage(borders, "Source Image Segmentation", -1);

		// INTERMEDIATE RESULT
		cv::imwrite("regions_src.png", borders);
		borders = UTIL::drawLabelBorders(pSeg->m_labels, app.srcImg, COLOR_BLACK);
		cv::imwrite("segments_src.png", borders);
		for (int i = 0; i < pSeg->m_intermediateRegions.size(); i++) {
			borders = UTIL::drawLabelBorders(pSeg->m_intermediateRegions[i], app.srcImg, COLOR_BLACK);
			cv::imwrite("intermediate_regions_src_" + to_string(i) + ".png", borders);
		}

		// run detection with current parameters
		Mat shadows;
		while (1) {
			pSpecDetection->DetectSourceShadows(app.sdc);
			Mat displayEdges = Mat(app.srcImg.size(), app.srcImg.type(), COLOR_WHITE);
			app.srcImg.copyTo(displayEdges, app.sdc.diffEdges);
			//displayEdges = UTIL::drawLabelBorders(pSeg->m_labels, displayEdges, COLOR_BLUE);
			displayEdges = UTIL::drawLabelBorders(pSeg->m_regions, displayEdges, COLOR_BLACK, 3);
			DisplayImage(displayEdges, win1, -1);

			shadows = Mat(app.srcImg.size(), CV_8UC3, Scalar(255, 255, 255));
			app.srcImg.copyTo(shadows, app.sdc.shadows);
			DisplayImage(shadows, win2, -1);
			if (!waitForAction(changed)) break;
		}
		// INTERMEDIATE RESULT
		cv::imwrite("shadow_src.png", shadows);

		pSpecDetection->m_srcShadows = app.sdc.shadows;
		destroyWindow("Source Image Segmentation");
	}	
}


/// parallel computation of results

/*	Thread safe result buffer access,
	set pending
*/
void App::ResultPending(int ind) {
	result_mtx->lock();
	results[ind].status = PENDING;
	result_mtx->unlock();
}

/*	Thread safe result buffer access, 
	set result
*/
void App::SetResult(int ind, Mat result, Mat blendingBorder, Mat correctionArea) {
	result_mtx->lock();
	results[ind] = Result(READY, result, blendingBorder, correctionArea);
	result_mtx->unlock();
}

/*	Thread safe result buffer access,
	get result image
*/
Mat App::GetCurrentResult() {
#ifdef DETECTION_SINGLE_RESULT
	return ComputeSingleResult();
#endif
	Mat result;
	app.result_mtx->lock();
	if(app.results[app.currInd].status == READY) {
		result = app.results[app.currInd].result;
		DisplayImage(app.results[app.currInd].blendingBorder, "blending border", -1);
		DisplayImage(app.results[app.currInd].correctionArea, "correction area", 0);
	}
	else result = Mat(app.tarImg.size(), CV_8U, Scalar(0));
	app.result_mtx->unlock();
	return result;
}

/*	Thread safe, start new thread to compute result
	if it is still empty in the result buffer
*/
void App::FillResultBuffer()
{
	mtx->lock();

	int curr = GetThreshFromResultIndex(app.currInd);
	int ind = numeric_limits<int>::max();
	// find closest empty result
	for (int i = 0; i < results.size(); i++) {
		if (results[i].status == EMPTY
			&& abs(curr - i) < abs(curr - ind))
			ind = i;
	}

	// stop when no empty results left 
	if (ind == numeric_limits<int>::max()) return;

	// get correction region for threshold and compute 
	// result in new thread
	ResultPending(ind);
	int th = GetThreshFromResultIndex(ind);
	pSpecDetection->DetectSpecularities(th);

	// create an own SpecularityRemoval object for each thread
	shared_ptr<SpecularityRemoval> pRemoval = make_shared<SpecularityRemoval>(
		SpecularityRemoval(pSpecDetection));
	thread thr([pRemoval, ind]()
	{
		Mat blendingBorder, correctionArea;
		Mat result = pRemoval->MakeCorrectedImage(blendingBorder, correctionArea);
		app.SetResult(ind, result, blendingBorder, correctionArea);
		// try to start next thread
		app.FillResultBuffer();
	});
	running_threads.push_back(shared_ptr<thread>(&thr));
	thr.detach();

	mtx->unlock();
}

/*	Run specularity detection for the different thresholds,
	compute the initial result and start new threads to 
	compute the other results in parallel 
*/
void App::AllocateResources() {
	results = vector<Result>();

	// determine first threshold and threshold range
	int th = pSpecDetection->GetInitialThreshold();
	app.lowerTh = th - DIST_TH_RANGE;
	app.upperTh = th + DIST_TH_RANGE;

	// initialize buffer
	int size = (DIST_TH_RANGE * 2) / DIST_TH_STEP + 1;
	for (int i = 0; i < size; i++)
		app.results.push_back(Result(EMPTY));

	app.currInd = app.GetResultIndexFromThresh(th);
}

Mat App::ComputeSingleResult() {
	int th = app.GetThreshFromResultIndex(app.currInd);
	pSpecDetection->DetectSpecularities(th);
	SpecularityRemoval removal(pSpecDetection);
	Mat detectionResult;
	app.tarImg.copyTo(detectionResult, removal.m_areaToCorrect);
	cv::imwrite("detection_tar.png", detectionResult);
	Mat blendingBorder, correctionArea;
	Mat result = removal.MakeCorrectedImage(blendingBorder, correctionArea);	
	DisplayImage(blendingBorder, "blending border", -1);
	DisplayImage(correctionArea, "correction area", -1);
	DisplayImage(detectionResult, "detectionResult", -1);
	DisplayImage(app.tarImg, "app.tarImg", -1);
	DisplayImage(app.srcImg, "app.srcImg", 100);
	
	return result;
}

Mat App::ComputeResults()
{
	cout << endl << "Running specularity detection" << endl;
	//app.scSpec = App::SelectSegmentComparison();
	//cout << "Selected method: " << App::scToString(app.scSpec) << endl;

	app.AllocateResources();

#ifdef DETECTION_SINGLE_RESULT
	return ComputeSingleResult();
#endif

	int ind = app.currInd;
	app.ResultPending(ind);

	// start new threads
	for (int i = 0; i < (NUM_THREADS - 1); i++)
		app.FillResultBuffer();

	// compute initial result in main thread
	app.mtx->lock();
	int th = app.GetThreshFromResultIndex(ind);
	pSpecDetection->DetectSpecularities(th);
	SpecularityRemoval removal(pSpecDetection);
	app.mtx->unlock();
	Mat blendingBorder, correctionArea;
	Mat result = removal.MakeCorrectedImage(blendingBorder, correctionArea);
	app.SetResult(ind, result, blendingBorder, correctionArea);
	
	return app.GetCurrentResult();
}

/*	Increase threshold and return result	
*/
Mat App::IncreaseThreshold()
{	
	if (app.currInd != (app.results.size() - 1)) 
		App::ConsolePrint("Current threshold: " +
			to_string(app.GetThreshFromResultIndex(
				++app.currInd)));
	
	return app.GetCurrentResult();
}

/*	Decrease threshold and return result
*/
Mat App::DecreaseThreshold()
{
	if (app.currInd != 0) {
		App::ConsolePrint("Current threshold: " +
			to_string(app.GetThreshFromResultIndex(
				--app.currInd)));
	}
	return app.GetCurrentResult();
}

/// tracking during object rectification 

/*	Initialize the tracking context
*/
vector<Point2i> App::StartTracking(Mat &img)
{
	// reset context
	app.mtc.win = "Rectify Image";
	app.mtc.img = img.clone();
	app.mtc.toggle = 0;
	app.mtc.pts.clear();

	DisplayImage(app.mtc.img, app.mtc.win, 500);
	setMouseCallback(app.mtc.win, MouseTracker);

	// wait for 4 points
	while (app.mtc.pts.size() != 4) waitKey(0);

	return app.mtc.pts;
}

/*	Callback for mouse events to handle 
	user action in the window
*/
void App::MouseTracker(int e, int x, int y, int d, void* args)
{
	int activeRadius = 40.0 / app.displayingScale;
	// translate from window to image coordinates
	Point2i clicked((double)x / app.displayingScale,
						(double)y / app.displayingScale);

	switch(e){
		// toggle navigation if there is a point close to clicked position
		case CV_EVENT_LBUTTONDOWN :{
			app.mtc.toggle = 1;
			vector<Point2i> pts = app.mtc.pts;
		
			// find closest point 
			float minDist = numeric_limits<float>::max();
			int ind = -1;
			for (int i = 0; i < pts.size(); i++) {
				float dist = sqrt((float)((pts[i].x - clicked.x)*(pts[i].x - clicked.x)
					+ (pts[i].y - clicked.y)*(pts[i].y - clicked.y)));

				if (dist < minDist) {
					minDist = dist;
					ind = i;
				}
			}

			// quit if point is out of active radius
			if (minDist > activeRadius) {
				app.mtc.toggle = 0;
				return;
			}

			// clear selected point from list
			app.mtc.pts.clear();
			for (int i = 0; i < pts.size(); i++) {
				if (i == ind) continue;
				app.mtc.pts.push_back(pts[i]);
			}
			// track selected point
			app.mtc.selPt = pts[ind];
		break; }

		// update point position during point navigation 
		case CV_EVENT_MOUSEMOVE : {
			// return if navigation is not toggled
			if (!app.mtc.toggle) return;
			// dont update point position if out of image border
			if (clicked.x < 0 || clicked.x > app.mtc.img.cols) return;
			if (clicked.y < 0 || clicked.y > app.mtc.img.rows) return;

			app.mtc.selPt = clicked;
			int radius = 5.0 / app.displayingScale;
			int thickness = 2.0 / app.displayingScale;
			// draw current points
			Mat drawImg = app.mtc.img.clone();
			for (Point2i pt : app.mtc.pts)
				circle(drawImg, pt, radius, Scalar(255, 0, 0), thickness);
	
			circle(drawImg, app.mtc.selPt, radius, Scalar(255, 0, 0), thickness);
			DisplayImage(drawImg, app.mtc.win, -1);
		break; }

		// navigation finished or new point
		case CV_EVENT_LBUTTONUP : {
			// if navigation is toggled store final point position
			if (app.mtc.toggle) {
				app.mtc.pts.push_back(app.mtc.selPt);
				app.mtc.toggle = 0;
				return;
			}

			// add clicked point if navigation is not toggled
			if (app.mtc.pts.size() >= 4) return;

			app.mtc.pts.push_back(clicked);

			int radius = 5.0 / app.displayingScale;
			int thickness = 2.0 / app.displayingScale;
			Mat drawImg = app.mtc.img.clone();
			for (Point2i pt : app.mtc.pts) 
				circle(drawImg, pt, radius, Scalar(255, 0, 0), thickness);
		
			DisplayImage(drawImg, app.mtc.win, -1);
		break; }

		// right click to remove points
		case CV_EVENT_RBUTTONDOWN : {
			app.mtc.pts.clear();
			DisplayImage(app.mtc.img, app.mtc.win, -1);
		break; }

		default: break;
	}
}



#ifdef EXPERIMENTAL
void App::ShowRegionSimilarities() {
	for (int i = 0; i < 2; i++) {
		ImageSegmentation* pSeg;	
		if (i == 0) pSeg = pSpecDetection.m_pTarSrcSegmentation;
		else pSeg = pSpecDetection.m_pSrcTarSegmentation;
		int num = pSeg->m_numRegions;
		Mat regions = pSeg->m_regions;
		Mat distBGR = pSeg->m_regionDistBGR;
		Mat distLAB = pSeg->m_regionDistLAB;
		for (int r = 0; r < num; r++) {
			Mat showRegionTar(regions.size(), CV_8UC3, Scalar(0, 0, 0)),
				showRegionSrc(regions.size(), CV_8UC3, Scalar(0, 0, 0));
			for (int y = 0; y < regions.rows; y++)
				for (int x = 0; x < regions.cols; x++) {
					if (regions.at<int>(y, x) == r) {
						showRegionTar.at<Vec3b>(y, x) = app.tarImg.at<Vec3b>(y, x);
						showRegionSrc.at<Vec3b>(y, x) = app.srcImg.at<Vec3b>(y, x);
					}
				}

			if (i == 0) cout << "TAR TO SRC SEGMENTATION:" << endl;
			else cout << "SRC TO TAR SEGMENTATION:" << endl;
			cout << "Region: " << to_string(r) << endl;
			cout << "Diff BGR: " << to_string(distBGR.at<float>(r, 0)) << endl;
			cout << "Diff LAB: " << to_string(distLAB.at<float>(r, 0)) << endl;
			DisplayImage(showRegionTar, "Tar Region", -1);
			DisplayImage(showRegionSrc, "SrcRegion", 0);
		}
	}
}
#endif