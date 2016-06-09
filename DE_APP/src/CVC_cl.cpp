/*---------------------------------------------------------------------------
   CVC_cl.cpp - OpenCL Cost Volume Construction Code
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "CVC_cl.h"

CVC_cl::CVC_cl(cl_context* context, cl_command_queue* commandQueue, cl_device_id device,
				Mat l, const int d) : lImg_ref(l), maxDis(d),
				context(context), commandQueue(commandQueue)
{
    //printf("OpenCL Colours and Gradients method for Cost Computation\n");

    //OpenCL Setup
    program = 0;
    kernel = 0;

    //if (!createProgram(context, device, "assets/cvc_double_novector.cl", &program))
    if (!createProgram(*context, device, "assets/cvc_float_novector.cl", &program))
    //if (!createProgram(context, device, "assets/cvc_double.cl", &program))
    //if (!createProgram(context, device, "assets/cvc_float.cl", &program))
    {
        cleanUpOpenCL(NULL, NULL, program, kernel, NULL, 0);
        cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
    }

    kernel = clCreateKernel(program, "cvc", &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        cleanUpOpenCL(NULL, NULL, NULL, kernel, NULL, 0);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    width = (cl_int)lImg_ref.cols;
    height = (cl_int)lImg_ref.rows;
    channels = (cl_int)lImg_ref.channels();
    dispRange = (cl_int)maxDis;

    bufferSize_color = width * height * sizeof(cl_float) * channels;
    bufferSize_grad = width * height * sizeof(cl_float);
    bufferSize_costVol = width * height * maxDis * sizeof(cl_float);

    /* An event to associate with the Kernel. Allows us to retreive profiling information later. */
    event = 0;

    /* [Kernel size] */
    /*
     * Each instance of the kernel operates on a single pixel portion of the image.
     * Therefore, the global work size is the number of pixel.
     */
    //globalWorksize[0] = (size_t)width/4;
    globalWorksize[0] = (size_t)width;
    globalWorksize[1] = (size_t)height;
    globalWorksize[2] = (size_t)dispRange;
    /* [Kernel size] */
}
CVC_cl::~CVC_cl(void)
{
    /* Release OpenCL objects. */
    cleanUpOpenCL(NULL, NULL, NULL, kernel, NULL, 0);
}

int CVC_cl::buildCV(const Mat& lImg, const Mat& rImg, cl_mem *memoryObjects)
{
    CV_Assert( lImg.type() == CV_32FC3 && rImg.type() == CV_32FC3 );

	cvtColor(lImg, lGray, CV_RGB2GRAY );
	cvtColor(rImg, rGray, CV_RGB2GRAY );

    //Sobel filter to compute X gradient     <-- investigate Mali Sobel OpenCL kernel
    Sobel( lGray, lGrdX, CV_32F, 1, 0, 1 ); // ex time 16 -17ms
	Sobel( rGray, rGrdX, CV_32F, 1, 0, 1 ); // for both
	lGrdX += 0.5;
	rGrdX += 0.5;

	/* Map the input memory objects to host side pointers. */
	bool EnqueueMapBufferSuccess = true;
	//Two 3-channel 2D float buffers W*H*3
	cl_float *clbuffer_lImg = (cl_float*)clEnqueueMapBuffer(*commandQueue, memoryObjects[CVC_LIMG], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, bufferSize_color, 0, NULL, NULL, &errorNumber);
	EnqueueMapBufferSuccess &= checkSuccess(errorNumber);
	cl_float *clbuffer_rImg = (cl_float*)clEnqueueMapBuffer(*commandQueue, memoryObjects[CVC_RIMG], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, bufferSize_color, 0, NULL, NULL, &errorNumber);
	EnqueueMapBufferSuccess &= checkSuccess(errorNumber);

	//Two 1-channel 2D float buffers W*H
	cl_float *clbuffer_lGrdX = (cl_float*)clEnqueueMapBuffer(*commandQueue, memoryObjects[CVC_LGRDX], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, bufferSize_grad, 0, NULL, NULL, &errorNumber);
	EnqueueMapBufferSuccess &= checkSuccess(errorNumber);
	cl_float *clbuffer_rGrdX = (cl_float*)clEnqueueMapBuffer(*commandQueue, memoryObjects[CVC_RGRDX], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, bufferSize_grad, 0, NULL, NULL, &errorNumber);
	EnqueueMapBufferSuccess &= checkSuccess(errorNumber);
	if (!EnqueueMapBufferSuccess)
	{
	   cleanUpOpenCL(NULL, NULL, program, kernel, NULL, 0);
	   cerr << "Mapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
	}

    //printf("CVC_cl: Copying data to OpenCL memory space\n");
	memcpy(clbuffer_lImg, lImg.data, bufferSize_color);
	memcpy(clbuffer_rImg, rImg.data, bufferSize_color);
	memcpy(clbuffer_lGrdX, lGrdX.data, bufferSize_grad);
	memcpy(clbuffer_rGrdX, rGrdX.data, bufferSize_grad);

    int arg_num = 0;
    /* Setup the kernel arguments. */
    bool setKernelArgumentsSuccess = true;
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[CVC_LIMG]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[CVC_RIMG]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[CVC_LGRDX]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[CVC_RGRDX]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_int), &height));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_int), &width));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[CV_LCV]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[CV_RCV]));
    if (!setKernelArgumentsSuccess)
    {
		cleanUpOpenCL(NULL, NULL, NULL, kernel, NULL, 0);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    if(OCL_STATS) printf("CVC_cl: Running CVC Kernels\n");
    /* Enqueue the kernel */
    if (!checkSuccess(clEnqueueNDRangeKernel(*commandQueue, kernel, 3, NULL, globalWorksize, NULL, 0, NULL, &event)))
    {
        cleanUpOpenCL(NULL, NULL, NULL, kernel, NULL, 0);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* Wait for completion */
    if (!checkSuccess(clFinish(*commandQueue)))
    {
        cleanUpOpenCL(NULL, NULL, NULL, kernel, NULL, 0);
        cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* Print the profiling information for the event. */
    if(OCL_STATS) printProfilingInfo(event);
    /* Release the event object. */
    if (!checkSuccess(clReleaseEvent(event)))
    {
        cleanUpOpenCL(*context, *commandQueue, program, kernel, NULL, 0);
        cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    return 0;
}
