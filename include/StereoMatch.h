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

#define DISPLAY
#define DEBUG_APP

class StereoMatch
{
public:
	StereoMatch(int argc, const char *argv[], int gotOpenCLDev);
	~StereoMatch(void);

	void compute(float& de_time_ms);

//    int imgType;
	int de_mode;
	int MatchingAlgorithm;
	int error_threshold;
	cv::UMat display_container;

	//StereoSGBM Variables
	cv::Ptr<StereoSGBM> ssgbm;

private:
	//Variables
	bool end_de, recaptureChessboards, recalibrate;
	int gotOCLDev, media_mode;
	char cap_key;

	std::string left_img_filename;
	std::string right_img_filename;
	std::string gt_img_filename;

	//Display Variables
	cv::UMat leftInputImg, rightInputImg;
	cv::UMat leftDispMap, rightDispMap;
    cv::UMat gtDispMap, errDispMap;
    cv::UMat blankDispMap;

	//local disparity map containers
    cv::UMat lDispMap, rDispMap, eDispMap;

	//input values
    int maxDis;

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
	StereoCameraProperties camProps;
	double minVal, maxVal;
	double minVal_gt, maxVal_gt;
	cv::UMat imgDisparity16S;

	//StereoGIF Variables
	DispEst* SMDE;
	int num_threads;

	//Function prototypes
	int setCameraResolution(unsigned int height, unsigned int width);
	int stereoCameraSetup(void);
	int captureChessboards(void);
	int setupOpenCVSGBM(int, int);
	int inputArgParser(int argc, char *argv[]);
	int parse_cli(int argc, const char * argv[]);
};

#endif //STEREOMATCH_H
