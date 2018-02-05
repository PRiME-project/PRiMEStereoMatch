/*---------------------------------------------------------------------------
   StereoMatch.h - Stereo Matching Application Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#ifndef STEREOMATCH_H
#define STEREOMATCH_H

#include "ComFunc.h"
#include "StereoCalib.h"
#include "DispEst.h"
#include "args.hxx"

#define DE_VIDEO 1
#define DE_IMAGE 2

#define NO_MASKS 0
#define MASK_NONE 1
#define MASK_NONOCC 2
#define MASK_DISC 3

#define DISPLAY
//#define DEBUG_APP
#define DEBUG_APP_MONITORS

static std::vector<std::string> dataset_names = std::vector<std::string>{"Art", "Books", "Cones", "Dolls", "Laundry", "Moebius", "Teddy"};

class StereoMatch
{
public:
	StereoMatch(int argc, const char *argv[], int gotOpenCLDev);
	~StereoMatch(void);

	int de_mode;
	int MatchingAlgorithm;
	int error_threshold;
	int mask_mode;
	int media_mode;
	cv::Mat display_container;
	//StereoSGBM Variables
	cv::Ptr<StereoSGBM> ssgbm;

	void compute(float& de_time_ms);
	int set_filenames(std::string dataset_name);
	bool user_dataset;

private:
	//Variables
	bool end_de, recaptureChessboards, recalibrate;
	int gotOCLDev;
	char cap_key;

	std::string left_img_filename, right_img_filename;
	std::string gt_img_filename, mask_occl_filename, mask_disc_filename;
	std::mutex set_filename_m;
	bool data_set_update, ground_truth_data;
	int mask_mode_next;
	int scale_factor, scale_factor_next;

	//Display Variables
	cv::Mat leftInputImg, rightInputImg;
	cv::Mat leftDispMap, rightDispMap;
    cv::Mat gtDispMap, errDispMap;
    cv::Mat blankDispMap;

	//local disparity map containers
    cv::Mat lDispMap, rDispMap, eDispMap;
    cv::Mat errMask;

	//input values
    int maxDis;

    //stage & process time measurements
    double cvc_time, cvf_time, dispsel_time, pp_time;

    //Frame Holders & Camera object
	cv::Mat lFrame, rFrame, vFrame;

	VideoCapture cap;
	//Image rectification maps
	cv::Mat mapl[2], mapr[2];
	cv::Rect cropBox;
	cv::Mat lFrame_rec, rFrame_rec;
	cv::Mat gtFrame;

	//StereoSGBM Variables
	StereoCameraProperties camProps;
	double minVal, maxVal;
	double minVal_gt, maxVal_gt;
	cv::Mat imgDisparity16S;

	//StereoGIF Variables
	DispEst* SMDE;
	int num_threads;

	//Function prototypes
	int setCameraResolution(unsigned int height, unsigned int width);
	int stereoCameraSetup(void);
	int captureChessboards(void);
	int setupOpenCVSGBM(int, int);
	int parse_cli(int argc, const char * argv[]);
};

#endif //STEREOMATCH_H
