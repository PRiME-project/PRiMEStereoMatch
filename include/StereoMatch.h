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

#define DE_VIDEO 1
#define DE_IMAGE 2

//#define DISPLAY
#define DEBUG_APP

class StereoMatch
{
public:
	//Variables
	bool end_de, recaptureChessboards, recalibrate;
	int gotOCLDev, media_mode;
	char cap_key;

	std::string left_img_filename;
	std::string right_img_filename;
	std::string gt_img_filename;

	//Display Variables
	cv::UMat display_container;
	cv::UMat leftInputImg, rightInputImg;
	cv::UMat leftDispMap, rightDispMap;
    cv::UMat gtDispMap, errDispMap;
    cv::UMat blankDispMap;

	//local disparity map containers
    cv::UMat lDispMap, rDispMap, eDispMap;

	//input values
    int maxDis;
//    int imgType;
	int MatchingAlgorithm;
	int error_threshold;

    //stage & process time measurements
    double cvc_time, cvf_time, dispsel_time, pp_time;

    //Frame Holders & Camera object
	cv::UMat lFrame, rFrame, vFrame;
    cv::UMat lFrame_tmp;
    cv::UMat rFrame_tmp;

	VideoCapture cap;
	//Image rectification maps
	cv::UMat mapl[2], mapr[2];
	cv::Rect cropBox;
	cv::UMat lFrame_rec, rFrame_rec;
	cv::UMat gtFrame, gtFrameImg;

	//StereoSGBM Variables
	cv::Ptr<StereoSGBM> ssgbm;
	StereoCameraProperties camProps;
	double minVal, maxVal;
	double minVal_gt, maxVal_gt;
	cv::UMat imgDisparity16S;

	//StereoGIF Variables
	DispEst* SMDE;
	int de_mode;
	int num_threads;

	//Function prototypes
	int stereoCameraSetup(void);
	int captureChessboards(void);
	int setupOpenCVSGBM(int, int);
	int inputArgParser(int argc, char *argv[]);
	int updateFrameType(void);
	void compute(float& de_time_ms);
	StereoMatch(int argc, char *argv[], int gotOpenCLDev);
    ~StereoMatch(void);
};

#endif //STEREOMATCH_H
