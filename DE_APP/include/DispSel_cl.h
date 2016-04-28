/*---------------------------------------------------------------------------
   DispSel_cl.h - OpenCL Disparity Selection Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "common.h"
#include "image.h"

#define DOUBLE_MAX 1e10

class DispSel_cl
{
public:

    Mat lImg_ref;
	const int maxDis;

    //OpenCL Variables
	cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    cl_kernel kernel;
    cl_device_id device;
    unsigned int numberOfMemoryObjects;
    cl_mem memoryObjects[4];
    cl_int errorNumber;
    //cl_event event;

	cl_int width, height;
    size_t bufferSize_char, bufferSize_grad, bufferSize_costVol;
    cl_float* lcostVol_cl,  *rcostVol_cl; //Input buffers
    size_t globalWorksize[2];
    cl_char* ldispMap_cl, *rdispMap_cl; //Output buffers

	DispSel_cl(Mat l, const int d);
	~DispSel_cl(void);

	int CVSelect(Mat* lcostVol, Mat* rcostVol, Mat& ldispMap, Mat& rdispMap);
};

