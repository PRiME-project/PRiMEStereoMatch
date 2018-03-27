/*---------------------------------------------------------------------------
   StereoMatch.h - Stereo Matching Application Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
  ---------------------------------------------------------------------------*/
#ifndef STEREOMATCH_H
#define STEREOMATCH_H

#include "ComFunc.h"
#include "StereoCalib.h"
#include "DispEst.h"
#include "args.hxx"
#include <sys/sysinfo.h>

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

struct Resolution{
	unsigned int height;
	unsigned int width;
};

class StereoMatch
{
public:
	StereoMatch(int argc, const char *argv[], int gotOpenCLDev);
	~StereoMatch(void);

	int gotOCLDev;
	int de_mode;
	int MatchingAlgorithm;
	int error_threshold;
	int mask_mode;
	int media_mode;
	cv::Mat display_container;
	std::deque<double> frame_rates;
	bool user_dataset;

	//StereoSGBM Variables
	cv::Ptr<StereoSGBM> ssgbm;
	int sgbm_mode;
	std::mutex dispMap_m;

	//Stereo GIF Variables
	unsigned int subsample_rate = 4;

	//Methods
	int createSGBMThreads(void);
	int computeStereoGIF(float& de_time);
	int computeStereoSGBM(float& de_time);
	int calcAccuracy(void);
	int updateDataset(std::string dataset_name);

private:
	//Variables
	bool end_de, recaptureChessboards, recalibrate;
	char cap_key;

	std::string left_img_filename, right_img_filename;
	std::string gt_img_filename, mask_occl_filename, mask_disc_filename;
	std::string curr_dataset;
	std::mutex input_data_m;
	bool ground_truth_data;
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
    double cvc_time_avg, cvf_time_avg, dispsel_time_avg, pp_time_avg;
    unsigned int frame_count;

    //Frame Holders & Camera object
	cv::Mat lFrame, rFrame, vFrame;
	VideoCapture cap;
	std::mutex cap_m;

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
	std::vector<std::thread> sgbm_threads;

	//StereoGIF Variables
	DispEst* SMDE;
	int num_threads;

	//Method prototypes
    int sgbm_thread(int tid);
	int setCameraResolution(unsigned int height, unsigned int width);
	std::vector<Resolution> resolutionSearch(void);
	int stereoCameraSetup(void);
	int captureChessboards(void);
	int setupOpenCVSGBM(cv::Ptr<StereoSGBM>& ssgbm, int channels, int ndisparities);
	int updateDisplay(void);
	int parseCLI(int argc, const char * argv[]);
};

#endif //STEREOMATCH_H
