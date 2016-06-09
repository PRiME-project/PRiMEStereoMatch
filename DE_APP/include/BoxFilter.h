/*---------------------------------------------------------------------------
   BoxFilter.h - OpenCL Boxfilter Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "common.h"

class BoxFilter
{
public:

	int height, width, dispRange;

    //OpenCL Variables
    cl_context* context;
	cl_command_queue* commandQueue;
    cl_program program;
    cl_kernel kernel_bf;
    cl_int errorNumber;
    cl_event event;

    //cl_int height, width, radius, dispRange;
    size_t bufferSize_char, bufferSize_color, bufferSize_grad, bufferSize_costVol;
    size_t globalWorksize_bf[3];

	BoxFilter(cl_context* context, cl_command_queue* commandQueue, cl_device_id device,
				cl_int _height, cl_int _width, cl_int radius, cl_int _dispRange);
	~BoxFilter(void);

	int filter(cl_mem *cl_in, cl_mem *cl_out);
	int matMul(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out);
};
