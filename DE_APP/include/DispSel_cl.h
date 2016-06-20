/*---------------------------------------------------------------------------
   DispSel_cl.h - OpenCL Disparity Selection Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "common.h"

#define DOUBLE_MAX 1e5

class DispSel_cl
{
public:

    Mat lImg_ref;
	const int maxDis;

    //OpenCL Variables
    cl_context* context;
	cl_command_queue* commandQueue;
    cl_program program;
    cl_kernel kernel;
    cl_int errorNumber;
    cl_event event;

	cl_int width, height;
	size_t bufferSize_2D_C1C, bufferSize_2D_C1F, bufferSize_3D_C1F;
    size_t globalWorksize[2];

	enum buff_id {CVC_LIMG = 0, CVC_RIMG, CVC_LGRDX, CVC_RGRDX, CV_LCV, CV_RCV, DS_LDM, DS_RDM};

	DispSel_cl(cl_context* context, cl_command_queue* commandQueue, cl_device_id device, Mat l, const int d);
	~DispSel_cl(void);

	int CVSelect(cl_mem* memoryObjects, Mat& ldispMap, Mat& rdispMap);
};

