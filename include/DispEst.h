/*---------------------------------------------------------------------------
   DispEst.h - Disparity Estimation Class Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "CVC.h"
#include "CVC_cl.h"
#include "CVF.h"
#include "CVF_cl.h"
#include "DispSel.h"
#include "DispSel_cl.h"
#include "PP.h"
#include "oclUtil.h"
#include "fastguidedfilter.h"
//
// Top-level Disparity Estimation Class
//
class DispEst
{
public:
    DispEst(cv::Mat l, cv::Mat r, const int d, int t, bool ocl);
    ~DispEst(void);

    //DispSel
    cv::Mat lDisMap;
    cv::Mat rDisMap;

    //Public Methods
	int setInputImages(cv::Mat l, cv::Mat r);
	int setThreads(unsigned int newThreads);
	void setSubsampleRate(unsigned int newRate) {subsample_rate = newRate;};
	int printCV(void);

    int CostConst();
    int CostConst_CPU();
    int CostConst_GPU();

    int CostFilter();
    int CostFilter_CPU();
    int CostFilter_GPU();
    int CostFilter_FGF();


    int DispSelect_CPU();
    int DispSelect_GPU();

    int PostProcess_CPU();
    int PostProcess_GPU();

private:
    //Private Variable
    cv::Mat lImg;
    cv::Mat rImg;

    int hei;
    int wid;
    int maxDis;
    int threads;
    bool useOCL;
    unsigned int subsample_rate = 4;

	//CVC
    cv::Mat lGrdX;
    cv::Mat rGrdX;
	//CVC & CVF
//    Mat lcostVol_cvc;
//    Mat rcostVol_cvc;
    cv::Mat* lcostVol;
    cv::Mat* rcostVol;
    //CVF
//    Mat* lImg_rgb;
//    Mat* rImg_rgb;
//    Mat* mean_lImg;
//    Mat* mean_rImg;
//    Mat* var_lImg;
//    Mat* var_rImg;
    //PP
    cv::Mat lValid;
    cv::Mat rValid;

    CVC* constructor;
    CVF* filter;
    DispSel* selector;
    PP* postProcessor;

    CVC_cl* constructor_cl;
	CVF_cl* filter_cl;
    DispSel_cl* selector_cl;

	//OpenCL Variables
	cl_context context;
    cl_command_queue commandQueue;
    cl_device_id device;
    unsigned int numberOfMemoryObjects;
    cl_mem memoryObjects[12]; //OpenCL Memory Buffers
    cl_int errorNumber;
    cl_event event;

    cl_int width, height, channels;
	size_t bufferSize_2D_8UC1; //DispMap,
	size_t bufferSize_2D; //Img, Gray, GrdX,
	size_t bufferSize_3D; //costVol

    //Private Methods
    //None
};
