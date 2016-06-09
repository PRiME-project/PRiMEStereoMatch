/*---------------------------------------------------------------------------
   BoxFilter.cpp - OpenCL Boxfilter Code
  ---------------------------------------------------------------------------
   Editor: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "BoxFilter.h"

BoxFilter::BoxFilter(cl_context* context, cl_command_queue* commandQueue, cl_device_id device,
						cl_int _height, cl_int _width, cl_int radius, cl_int _dispRange) :
						height(_height), width(_width), dispRange(_dispRange), context(context), commandQueue(commandQueue)
{
    //fprintf(stderr, "OpenCL Guided Image Filtering method for Cost Computation\n" );

    //OpenCL Setup
    program = 0;
    kernel_bf = 0;

    //if (!createProgram(context, device, "assets/boxfilter_8x8.cl", &program))
    if (!createProgram(*context, device, "assets/boxfilter_vector16.cl", &program))
    //if (!createProgram(context, device, "assets/boxfilter_vector8.cl", &program))
    //if (!createProgram(context, device, "assets/boxfilter_novector.cl", &program))
    {
        cleanUpOpenCL(NULL, NULL, program, kernel_bf, NULL, 0);
        cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
    }
	kernel_bf = clCreateKernel(program, "boxfilter", &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        cleanUpOpenCL(NULL, NULL, program, kernel_bf, NULL, 0);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    /* An event to associate with the Kernel. Allows us to retreive profiling information later. */
    event = 0;

	//Buffers in accending size order
	bufferSize_char = width * height * sizeof(cl_char);
	bufferSize_grad = width * height * sizeof(cl_float);
	bufferSize_costVol = width * height * dispRange * sizeof(cl_float);

    /* [Kernel size] */
    globalWorksize_bf[0] = (size_t)width/16;
    //globalWorksize_bf[0] = (size_t)width/8;
    //globalWorksize_bf[0] = (size_t)width;
    globalWorksize_bf[1] = (size_t)height;
    globalWorksize_bf[2] = (size_t)dispRange;
}

BoxFilter::~BoxFilter(void)
{
	/* Release OpenCL objects. */
	cleanUpOpenCL(NULL, NULL, program, kernel_bf, NULL, 0);
}

int BoxFilter::filter(cl_mem *cl_in, cl_mem *cl_out)
{
	int arg_num = 0;
	/* Setup the kernel arguments. */
	bool setKernelArgumentsSuccess = true;
	setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_bf, arg_num++, sizeof(cl_mem), cl_in));
	setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_bf, arg_num++, sizeof(cl_int), &width));
	setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_bf, arg_num++, sizeof(cl_int), &height));
	setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_bf, arg_num++, sizeof(cl_mem), cl_out));

	if (!setKernelArgumentsSuccess)
	{
	   cleanUpOpenCL(*context, *commandQueue, program, kernel_bf, NULL, 0);
		cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
	}


	/* Enqueue the kernel */
	if (!checkSuccess(clEnqueueNDRangeKernel(*commandQueue, kernel_bf, 3, NULL, globalWorksize_bf, NULL, 0, NULL, &event)))
	{
	   cleanUpOpenCL(*context, *commandQueue, program, kernel_bf, NULL, 0);
		cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	/* Wait for completion */
	if (!checkSuccess(clFinish(*commandQueue)))
	{
	   cleanUpOpenCL(*context, *commandQueue, program, kernel_bf, NULL, 0);
		cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	//cout << "BoxFilter ";
	//printProfilingInfo(event);
//    /* Release the event object. */
//    if (!checkSuccess(clReleaseEvent(event)))
//    {
//        cleanUpOpenCL(context, commandQueue, program, kernel_bf, memoryObjects, numberOfMemoryObjects);
//        cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
//        return 1;
//    }

    return 0;
}
