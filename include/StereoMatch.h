/*---------------------------------------------------------------------------
   StereoMatch.h - Stereo Matching Application Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "StereoCalib.h"
#include "DispEst.h"

class StereoMatch
{
public:
	//Variables
	bool end_de, recaptureChessboards, recalibrate, video;
	int gotOCLDev;
	char cap_key;

	string left_img_filename;
	string right_img_filename;
	string gt_img_filename;

	//Display Variables
	Mat display_container;
	Mat leftInputImg, rightInputImg;
	Mat leftDispMap, rightDispMap;
    Mat gtDispMap, errDispMap;
    Mat blankDispMap;

	//local disparity map containers
    Mat lDispMap, rDispMap, eDispMap;

	//input values
    int maxDis;
    int imgType;
	int MatchingAlgorithm;
	int error_threshold;

    //stage & process time measurements
    float cvc_time, cvf_time, dispsel_time, pp_time;

    //Frame Holders & Camera object
	Mat lFrame, rFrame, vFrame;
	VideoCapture cap;
	//Image rectification maps
	Mat mapl[2], mapr[2];
	Rect cropBox;
	Mat lFrame_rec, rFrame_rec;
	Mat gtFrame, gtFrameImg;

	//StereoSGBM Variables
	Ptr<StereoSGBM> ssgbm;
	StereoCameraProperties camProps;
	double minVal; double maxVal;
	double minVal_gt; double maxVal_gt;
	Mat imgDisparity16S;

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
	int Compute(float& de_time_ms);
	StereoMatch(int argc, char *argv[], int gotOpenCLDev);
    ~StereoMatch(void);
};
