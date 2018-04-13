/*---------------------------------------------------------------------------
   DispEst.h - Disparity Estimation Class Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "CVC.h"
#include "CVF.h"
#include "DispSel.h"
#include "PP.h"
#include "fastguidedfilter.h"
//
// Top-level Disparity Estimation Class
//
class DispEst
{
public:
    DispEst(cv::Mat l, cv::Mat r, const int d, int t);
    ~DispEst(void);

    //DispSel
    cv::Mat lDisMap;
    cv::Mat rDisMap;

    //Public Methods
	int setInputImages(cv::Mat l, cv::Mat r);
	int setThreads(unsigned int newThreads);
	void setSubsampleRate(unsigned int newRate) {subsample_rate = newRate;};
	int printCV(void);

    int CostConst_OMP();
    int CostFilter_FGF_OMP();
    int DispSelect_CPU();
    int PostProcess_CPU();

private:
    //Private Variable
    __attribute__((target(mic))) cv::Mat lImg;
    __attribute__((target(mic))) cv::Mat rImg;

    int hei;
    int wid;
    int maxDis;
    int threads;
    unsigned int subsample_rate;

	//CVC
    __attribute__((target(mic))) cv::Mat lGrdX;
    __attribute__((target(mic))) cv::Mat rGrdX;
	//CVC & CVF
    __attribute__((target(mic))) cv::Mat* lcostVol;
    __attribute__((target(mic))) cv::Mat* rcostVol;
    //CVF
    //PP
    cv::Mat lValid;
    cv::Mat rValid;

    __attribute__((target(mic))) CVC* constructor;
    __attribute__((target(mic))) CVF* filter;
    __attribute__((target(mic))) DispSel* selector;
    __attribute__((target(mic))) PP* postProcessor;

    //Private Methods
    //None
};
