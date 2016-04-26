/*---------------------------------------------------------------------------
   BoxFilter.h - OpenCL Boxfilter Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "common.h"
#include "image.h"

class BoxFilter
{
public:

    //OpenCL Variables
	cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    cl_kernel kernel;
    cl_device_id device;
    unsigned int numberOfMemoryObjects;
    cl_mem memoryObjects[2] = {0, 0};
    cl_int errorNumber;
    cl_event event;

    cl_int height, width, radius, dispRange;
    size_t bufferSize;
    size_t globalWorksize[3];

	BoxFilter(int _height, int _width, int _radius, int _dispRange);
	~BoxFilter(void);

	int filter(Mat *in_ptr, Mat *out_ptr);
};
