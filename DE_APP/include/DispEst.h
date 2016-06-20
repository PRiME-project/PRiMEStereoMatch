/*---------------------------------------------------------------------------
   DispEst.h - Disparity Estimation Class Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "CVC.h"
#include "CVC_cl.h"
#include "CVF.h"
#include "CVF_cl.h"
#include "DispSel.h"
#include "DispSel_cl.h"
#include "PP.h"
#include "common.h"

//
// Overarching Disparity Estimation Class
//
class DispEst
{
public:

    Mat lImg;
    Mat rImg;

    int hei;
    int wid;
    int maxDis;
    int threads;
    bool useOCL;

	//CVC & CVF
    Mat* lcostVol;
    Mat* rcostVol;
    //CVF
    Mat* lImg_rgb;
    Mat* rImg_rgb;
    Mat* mean_lImg;
    Mat* mean_rImg;
    Mat* var_lImg;
    Mat* var_rImg;
    //DispSel
    Mat lDisMap;
    Mat rDisMap;
    //PP
    Mat lValid;
    Mat rValid;

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
    cl_mem memoryObjects[8]; //OpenCL Memory Buffers
    cl_int errorNumber;
    cl_event event;

    cl_int width, height, channels;
	size_t bufferSize_2D_C1C, bufferSize_2D_C1F, bufferSize_2D_C3F, bufferSize_3D_C1F;

	enum buff_id {CVC_LIMG = 0, CVC_RIMG, CVC_LGRDX, CVC_RGRDX, CV_LCV, CV_RCV, DS_LDM, DS_RDM};

    DispEst(Mat l, Mat r, const int d, const int t, bool ocl);
    ~DispEst(void);

	void printCV(void);

    void CostConst();
    void CostConst_CPU();
    void CostConst_GPU();

    void CostFilter();
    void CostFilter_CPU();
    void CostFilter_GPU();

    void DispSelect_CPU();
    void DispSelect_GPU();

    void PostProcess_CPU();
};
