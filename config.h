#pragma once

#include <string>

/*	This file contains some parameters that can be adjusted
*/

const int MAX_IMAGE_SIZE = 600;
const std::string	TARGET_IMAGE_PATH = "C:/Users/Alexandra/Desktop/2nd semester/Project/Images/bird/bird2.jpg",
					SOURCE_IMAGE_PATH = "C:/Users/Alexandra/Desktop/2nd semester/Project/Images/bird/bird3.jpg";

const int NUM_THREADS = 4;

// IO
const int MAX_DISPLAYING_SIZE = 600;
const int	BUTTON_OK = 13,
			BUTTON_PLUS = 43, 
			BUTTON_MINUS = 45,
			BUTTON_ONE = 49,
			BUTTON_TWO = 50,
			BUTTON_THREE = 51,
			BUTTON_ZERO = 57,
			BUTTON_SAVE = 115;

const cv::Scalar	COLOR_RED = cv::Scalar(0, 0, 255),
					COLOR_BLUE = cv::Scalar(255, 0, 0),
					COLOR_WHITE = cv::Scalar(255, 255, 255),
					COLOR_BLACK = cv::Scalar(0, 0, 0);

#define DETECTION_SINGLE_RESULT

// segmentation parameters
const int	MAX_NUM_SEGMENTS = 4000,
			SEED_ITERATIONS = 16;

//#define MERGING_USE_INTENSITY_DIFF_BGR
const cv::Vec3i	MERGE_BORDER_TH(10, 10, 10);
const double	MERGE_INTENSITY_DIFF_BGR = 4.0;

//#define MERGING_USE_DIST_BGR
const double	MERGE_DIST_BGR = 4.0;

#define MERGING_USE_DIST_LAB
const double	MERGE_DIST_LAB = 4.0;

const double	MIN_REGION_SIZE_FRACTION = 0.005; //0.005;


// mask post processing parameters
const double MAX_BLOB_SIZE_FRACTION = 0.002, MAX_HOLE_SIZE_FRACTION = 0.005;
const int	SHADOW_DILATION_ITERATIONS = 10,
			CORRECTION_MASK_DILATION_ITERATIONS = 10,
			BORDER_SMOOTHING_ITERATIONS = 2;

// specularity threshold 
const int	DIST_TH_RANGE = 200, 
			DIST_TH_STEP = 5;
const double	INITIAL_TH_FRACTION = 0.75, 
				INITIAL_TH_SAMPLES_PERCENTAGE = 0.75;

//#define EXPERIMENTAL
//#define MERGE_TWICE
const double MERGE_TWICE_DIST_FACTOR = 2.0;

//#define MERGE_TRIPLE
const double MERGE_TRIPLE_DIST_FACTOR = 4.0;