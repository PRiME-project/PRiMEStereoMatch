/*---------------------------------------------------------------------------
   DispSel_cl.h - OpenCL Disparity Selection Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "oclUtil.h"

#define FILE_DS_PROG BASE_DIR "assets/dispsel.cl"

class DispSel_cl
{
public:
//	int imgType;
	const int maxDis;

    //OpenCL Variables
    cl_context* context;
	cl_command_queue* commandQueue;
    cl_program program;
    char kernel_name[128];
    cl_kernel kernel;
    cl_int errorNumber;
    cl_event event;

	cl_int width, height;
	size_t bufferSize_2D_8UC1, bufferSize_3D_8UC1;
    size_t globalWorksize[2];

	DispSel_cl(cl_context* context, cl_command_queue* commandQueue, cl_device_id device, Mat* I, const int d);
	~DispSel_cl(void);

	int CVSelect(cl_mem* memoryObjects, Mat& ldispMap, Mat& rdispMap);
};

