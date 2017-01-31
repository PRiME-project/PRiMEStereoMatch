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

#define BASE_DIR "../"
#define FILE_INTRINSICS 	BASE_DIR "data/intrinsics.yml"
#define FILE_EXTRINSICS 	BASE_DIR "data/extrinsics.yml"

// Recalibrate
#define FILE_CALIB_XML  	BASE_DIR "data/stereo_calib.xml"
// Recapture
#define FILE_TEMPLATE_LEFT	BASE_DIR "data/chessboard%dL.png"
#define FILE_TEMPLATE_RIGHT	BASE_DIR "data/chessboard%dR.png"

class StereoMatch
{
public:
	//Variables
	bool end_de, recaptureChessboards, recalibrate, video;
	int gotOCLDev;
	char cap_key;

	int MatchingAlgorithm;
	Ptr<StereoSGBM> ssgbm;
	StereoCameraProperties camProps;

	Mat display_container;
	Mat leftInputImg, rightInputImg;
	Mat leftDispMap, rightDispMap;
    Mat lDispMap, rDispMap;

	//input values
    int maxDis;
    int imgType;

    //stage & process time measurements
    float cvc_time, cvf_time, dispsel_time, pp_time;
    float de_time;

    //Frame Holders & Camera object
	Mat lFrame, rFrame, vFrame;
	VideoCapture cap;
	//Image rectification maps
	Mat mapl[2], mapr[2];
	Rect cropBox;
	Mat lFrame_rec, rFrame_rec;
	//StereoSGBM Variables
	double minVal; double maxVal;
	Mat imgDisparity16S;
	//StereoGIF Variables
	DispEst* SMDE;
	int de_mode;
	int num_threads;

	char left_img_filename[100];
	char right_img_filename[100];

	//Function prototypes
	int stereoCameraSetup(void);
	int captureChessboards(void);
	int setupOpenCVSGBM(int, int);
	int inputArgParser(int argc, char *argv[]);
	int imgTypeChange(int type);
	int Compute(void);
	StereoMatch(int argc, char *argv[], int gotOpenCLDev);
    ~StereoMatch(void);
};
