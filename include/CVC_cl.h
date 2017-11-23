/*---------------------------------------------------------------------------
   CVC_cl.h - OpenCL Cost Volume Construction Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "oclUtil.h"

#define FILE_CVC_PROG BASE_DIR "assets/cvc.cl"

//
// TAD + GRD for Cost Computation
//
class CVC_cl
{
public:

    //Data Variables
    Mat lGray, rGray;
	Mat lGrdX, rGrdX;
	int maxDis;

	//OpenCL Variables
    cl_context* context;
	cl_command_queue* commandQueue;
    cl_program program;
    char kernel_name[128];
    cl_kernel kernel;
    cl_int errorNumber;
    cl_event event;

    cl_int width, height, channels;
    size_t bufferSize_2D, bufferSize_3D;
    size_t globalWorksize[3];

    CVC_cl(cl_context* context, cl_command_queue* commandQueue, cl_device_id device, Mat* I, const int d);
    ~CVC_cl(void);

	int buildCV(const Mat& lImg, const Mat& rImg, cl_mem* memoryObjects);
};
