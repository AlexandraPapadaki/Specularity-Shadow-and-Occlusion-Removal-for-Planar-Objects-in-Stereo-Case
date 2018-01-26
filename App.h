#pragma once

#include <iostream>
#include <mutex>

#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

#include "config.h"

using namespace std;
using namespace cv;


/*	Static class of the application, which handles
	user interaction and manages the application state
*/
class App
{
public:
	static void InitApp(Mat &tar, Mat &src);
	static void ConsolePrint(std::string msg);
	static void DisplayImage(Mat &img, std::string win, int ms);

	static void RunAlignment(bool rectification);
	static vector<Point2i> StartTracking(Mat &img);
	static void MouseTracker(int e, int x, int y, int d, void* args);

	static void SetupAlgorithm(bool tarHasShadows, bool srcHasShadows);
	
	static void RunShadowDetection();
	static Mat ComputeSingleResult();
	static Mat ComputeResults();
	static Mat IncreaseThreshold();
	static Mat DecreaseThreshold();
#ifdef EXPERIMENTAL
	static void ShowRegionSimilarities();
#endif

	// perform segment comparison during shadow and specularity 
	// detection based on the following criteria
	enum SegmentComparison {
		INTENSITY_DIFF_RGB,
		DIST_RGB,
		DIST_LAB,
		L_DIFF_LAB,
		GRAY_DIFF
	};
	SegmentComparison scSpec = GRAY_DIFF, scShadows = GRAY_DIFF;
	static SegmentComparison getScSpec();
	static SegmentComparison getScShadows();

	static string scToString(SegmentComparison sc) {
		string scStr[] = { 
			"RGB INTENSITY DIFFERENCE", "RGB EUCLIDEAN DISTANCE",
			"LAB EUCLIDEAN DISTANCE", "LIGHTNESS DIFFERENCE USING LAB",
			"GRAY VALUE DIFFERENCE"
		};	
		return scStr[sc];
	}
	static SegmentComparison SelectSegmentComparison() {
		cout << "Enter method for segment comparison:" << endl;
		cout << scToString(INTENSITY_DIFF_RGB) << " (RGB1)" << endl;
		cout << scToString(DIST_RGB) << " (RGB2)" << endl;
		cout << scToString(DIST_LAB) << " (LAB)" << endl;
		cout << scToString(L_DIFF_LAB) << " (L)" << endl;
		cout << scToString(GRAY_DIFF) << " (GRAY)" << endl;

		while (1) {
			string method;
			cin >> method;
			if (method == "RGB1" || method == "rgb1") return INTENSITY_DIFF_RGB;
			if (method == "RGB2" || method == "rgb2") return DIST_RGB;
			if (method == "LAB" || method == "lab") return DIST_LAB;
			if (method == "L" || method == "l") return L_DIFF_LAB;
			if (method == "GRAY" || method == "gray") return GRAY_DIFF;

			cout << " < " << method << " > is not a valid method!" << endl;			
		}
	}

	/// thread safe result buffer access
	void ResultPending(int ind);
	void SetResult(int ind, Mat result, Mat blendingBorder, Mat correctionArea);
	static Mat GetCurrentResult();

	// needed for tracking during object rectification
	struct MouseTrackingContext {
		std::string win;
		Mat img;
		vector<Point2i> pts;
		bool toggle;
		Point2i selPt;
	};

	// shadow detection context with current parameters and results
	struct ShadowDetectionContext {
		// parameters
		int minInitiallyDetected;
		int minPercentageDetected;
		int gradTh;

		// results
		Mat diffEdges;
		Mat shadows;
		
		void ResetParams() {
			minInitiallyDetected = 20; // 20%
			minPercentageDetected = 50; // 50%
			gradTh = 10;
		}
		ShadowDetectionContext() { ResetParams(); }
	};

	// final results which are pre computed for different thresholds
	enum ResultState { EMPTY, READY, PENDING };
	struct Result {
		Mat result;
		Mat blendingBorder;
		Mat correctionArea;
		ResultState status;
		Result(ResultState st, Mat res = Mat(), 
				Mat blend = Mat(), Mat cor = Mat()) { 
			status = st; 
			result = res; 
			blendingBorder = blend;
			correctionArea = cor;
		}
	};
	ShadowDetectionContext sdc;

private:
	// app data
	double displayingScale = 1.0;
	Mat tarImg, srcImg;
	MouseTrackingContext mtc;
	
	// threshold boundary for adjustment
	int currInd;
	int lowerTh, upperTh;
	// buffer and precompute results to hide delay
	vector<Result> results;
	shared_ptr<mutex>	mtx = shared_ptr<mutex>(new mutex()), 
						result_mtx = shared_ptr<mutex>(new mutex()), 
						console_mtx = shared_ptr<mutex>(new mutex());
	vector<shared_ptr<thread>> running_threads;
	void AllocateResources();

	// some internal functions
	int GetResultIndexFromThresh(int th) {
		return results.size()/2 + ( th - ((upperTh + lowerTh)/2) )/DIST_TH_STEP;
	}

	int GetThreshFromResultIndex(int ind) {
		return (upperTh + lowerTh)/2 + (ind - results.size()/2)*DIST_TH_STEP;
	}
	
	void FillResultBuffer();
};

