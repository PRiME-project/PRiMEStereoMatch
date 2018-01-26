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
#include <mutex>
#include <deque>

#define DE_VIDEO 1
#define DE_IMAGE 2

#define DISPLAY
//#define DEBUG_APP

class StereoMatch
{
public:
	StereoMatch(int argc, char *argv[], int gotOpenCLDev);
    ~StereoMatch(void);

    int sgbm_thread(std::mutex &cap_m, std::mutex &dispMap_m, bool &end_de);
    int display_thread(std::mutex &dispMap_m);

	//StereoGIF Variables
	int de_mode;
	int num_threads;
	int MatchingAlgorithm;
	int error_threshold;
	cv::UMat display_container;
	std::deque<double> frame_rates;

private:
	//Variables
	bool end_de, recaptureChessboards, recalibrate;
	int gotOCLDev, media_mode;
	char cap_key;

	std::string left_img_filename;
	std::string right_img_filename;
	std::string gt_img_filename;

    //Frame Holders & Camera object
	cv::UMat lFrame, rFrame, vFrame;
    cv::UMat lFrame_tmp, rFrame_tmp;
	cv::UMat lFrame_rec, rFrame_rec;

	//Display Variables
	cv::UMat leftInputImg, rightInputImg;
	cv::UMat leftDispMap, rightDispMap;
	cv::UMat lDispMap, rDispMap, eDispMap;
	cv::UMat errDispMap;

    cv::UMat gtDispMap;
    cv::UMat blankDispMap;

	//input values
    int maxDis;
	double minVal_gt, maxVal_gt;

    //stage & process time measurements
    double cvc_time, cvf_time, dispsel_time, pp_time;

	VideoCapture cap;
	//Image rectification maps
	cv::UMat mapl[2], mapr[2];
	cv::Rect cropBox;
	cv::UMat gtFrame, gtFrameImg;

	//StereoSGBM Variables
	StereoCameraProperties camProps;
	//StereoGIF Variables
	DispEst* SMDE;

	//Function prototypes
	int stereoCameraSetup(void);
	int captureChessboards(void);
	int setupOpenCVSGBM(cv::Ptr<StereoSGBM>& ssgbm, int channels, int ndisparities);
	int inputArgParser(int argc, char *argv[]);
};

#endif //STEREOMATCH_H
