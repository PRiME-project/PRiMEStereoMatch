/*---------------------------------------------------------------------------
   StereoMatch.h - Stereo Matching Application Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "StereoCalib.h"
#include "DispEst.h"

class StereoMatch
{
public:
	//Variables
	bool end_de, recaptureChessboards, recalibrate, video, gotOCLDev;
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
    //stage & process time measurements
    timespec realtime;
    float cvc_time, cvf_time;
    float cvc_start, cvc_end, cvf_start, cvf_end;
    float dispsel_time, pp_time;
    float dispsel_start, dispsel_end, pp_start, pp_end;
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
	bool filter;

	char left_img_filename[100];
	char right_img_filename[100];
//	char *left_img_filename = "data/view1.png";
//	char *right_img_filename = "data/view5.png";

	//Function prototypes
	int Compute(void);
	StereoMatch(int argc, char *argv[], bool gotOpenCLDev);
    ~StereoMatch(void);
	int stereoCameraSetup(void);
	int captureChessboards(void);
	int setupOpenCVSGBM(int, int);
	void inputArgParser(int argc, char *argv[]);
};
