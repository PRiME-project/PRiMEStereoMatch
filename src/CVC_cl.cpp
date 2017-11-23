/*---------------------------------------------------------------------------
   CVC_cl.cpp - OpenCL Cost Volume Construction Code
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "CVC_cl.h"

CVC_cl::CVC_cl(cl_context* context, cl_command_queue* commandQueue, cl_device_id device,
				Mat* I, const int d) : maxDis(d),
				context(context), commandQueue(commandQueue)
{
    //printf("OpenCL Colours and Gradients method for Cost Computation\n");

    //OpenCL Setup
    program = 0;
    kernel = 0;
//    imgType = I->type() & CV_MAT_DEPTH_MASK;

    if (!createProgram(*context, device, FILE_CVC_PROG, &program))
    {
        cleanUpOpenCL(NULL, NULL, program, NULL, NULL, 0);
        cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
    }

    width = (cl_int)I->cols;
    height = (cl_int)I->rows;
    channels = (cl_int)I->channels();

//	if(imgType == CV_32F)
//	{
		strcpy(kernel_name, "cvc_float_nv");
		//strcpy(kernel_name, "cvc_float_v4");

		bufferSize_2D = width * height * sizeof(cl_float);
		bufferSize_3D = width * height * maxDis * sizeof(cl_float);

		//cvc_uchar_nv
		globalWorksize[0] = (size_t)width;
		//cvc_uchar_v4
//		globalWorksize[0] = (size_t)width/4;

		globalWorksize[1] = (size_t)height;
		globalWorksize[2] = (size_t)maxDis;
//	}
//	else if(imgType == CV_8U)
//	{
//		strcpy(kernel_name, "cvc_uchar_vx");
//		//strcpy(kernel_name, "cvc_uchar_v16");
//		//strcpy(kernel_name, "cvc_uchar_nv");
//
//		bufferSize_2D = width * height * sizeof(cl_uchar);
//		bufferSize_3D = width * height * maxDis * sizeof(cl_uchar);
//
//		//cvc_uchar_vx
//	    globalWorksize[0] = (size_t)height;
//	    globalWorksize[1] = (size_t)1;
//		//cvc_uchar_v16
////		globalWorksize[0] = (size_t)width/16;
//		//cvc_uchar_nv
////		globalWorksize[0] = (size_t)width;
////		globalWorksize[1] = (size_t)height;
//
//		globalWorksize[2] = (size_t)maxDis;
//
//	}
//    else{
//		printf("CVC_cl: Error - Unrecognised data type in processing! (CVC_cl)\n");
//		exit(1);
//    }
	kernel = clCreateKernel(program, kernel_name, &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        cleanUpOpenCL(NULL, NULL, NULL, NULL, NULL, 0);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
		exit(1);
    }
    else{
		printf("CVC_cl: OpenCL kernels created.\n");
	}

    /* An event to associate with the Kernel. Allows us to retreive profiling information later. */
    event = 0;
}
CVC_cl::~CVC_cl(void)
{
    /* Release OpenCL objects. */
    cleanUpOpenCL(NULL, NULL, NULL, kernel, NULL, 0);
}

int CVC_cl::buildCV(const Mat& lImg, const Mat& rImg, cl_mem *memoryObjects)
{
	Mat lImgRGB[channels], rImgRGB[channels];
    split(lImg, lImgRGB);
    split(rImg, rImgRGB);

	cvtColor(lImg, lGray, CV_RGB2GRAY);
	cvtColor(rImg, rGray, CV_RGB2GRAY);

	/* Map the input memory objects to host side pointers. */
	bool EnqueueMapBufferSuccess = true;
//	if(imgType == CV_32F)
//	{
	    //Sobel filter to compute X gradient     <-- investigate Mali Sobel OpenCL kernel
		Sobel( lGray, lGrdX, CV_32F, 1, 0, 1 ); // ex time 16 -17ms
		Sobel( rGray, rGrdX, CV_32F, 1, 0, 1 ); // for both
		lGrdX += 0.5;
		rGrdX += 0.5;

		cl_float *clbuffer_lImgRGB[3], *clbuffer_rImgRGB[3];
		for (int i = 0; i < channels; i++)
		{
			clbuffer_lImgRGB[i] = (cl_float*)clEnqueueMapBuffer(*commandQueue, memoryObjects[i], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, bufferSize_2D, 0, NULL, NULL, &errorNumber);
			EnqueueMapBufferSuccess &= checkSuccess(errorNumber);
			clbuffer_rImgRGB[i] = (cl_float*)clEnqueueMapBuffer(*commandQueue, memoryObjects[i+channels], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, bufferSize_2D, 0, NULL, NULL, &errorNumber);
			EnqueueMapBufferSuccess &= checkSuccess(errorNumber);

			memcpy(clbuffer_lImgRGB[i], lImgRGB[i].data, bufferSize_2D);
			memcpy(clbuffer_rImgRGB[i], rImgRGB[i].data, bufferSize_2D);
		}
//	}
//	else if(imgType == CV_8U)
//	{
//		//Sobel filter to compute X gradient
//		Sobel( lGray, lGrdX, CV_8U, 1, 0, 1 );
//		Sobel( rGray, rGrdX, CV_8U, 1, 0, 1 );
//		lGrdX += 0.5;
//		rGrdX += 0.5;
//
//		cl_uchar *clbuffer_lImgRGB[3], *clbuffer_rImgRGB[3];
//		//Six 1-channel 2D buffers W*H
//		for (int i = 0; i < channels; i++)
//		{
//			clbuffer_lImgRGB[i] = (cl_uchar*)clEnqueueMapBuffer(*commandQueue, memoryObjects[i], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, bufferSize_2D, 0, NULL, NULL, &errorNumber);
//			EnqueueMapBufferSuccess &= checkSuccess(errorNumber);
//			clbuffer_rImgRGB[i] = (cl_uchar*)clEnqueueMapBuffer(*commandQueue, memoryObjects[i+channels], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, bufferSize_2D, 0, NULL, NULL, &errorNumber);
//			EnqueueMapBufferSuccess &= checkSuccess(errorNumber);
//
//			memcpy(clbuffer_lImgRGB[i], lImgRGB[i].data, bufferSize_2D);
//			memcpy(clbuffer_rImgRGB[i], rImgRGB[i].data, bufferSize_2D);
//		}
//	}

	//Two 1-channel 2D buffers W*H
	cl_uchar *clbuffer_lGrdX = (cl_uchar*)clEnqueueMapBuffer(*commandQueue, memoryObjects[CVC_LGRDX], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, bufferSize_2D, 0, NULL, NULL, &errorNumber);
	EnqueueMapBufferSuccess &= checkSuccess(errorNumber);
	cl_uchar *clbuffer_rGrdX = (cl_uchar*)clEnqueueMapBuffer(*commandQueue, memoryObjects[CVC_RGRDX], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, bufferSize_2D, 0, NULL, NULL, &errorNumber);
	EnqueueMapBufferSuccess &= checkSuccess(errorNumber);
	if (!EnqueueMapBufferSuccess)
	{
	   cleanUpOpenCL(NULL, NULL, program, NULL, NULL, 0);
	   cerr << "Mapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
	}

    //printf("CVC_cl: Copying data to OpenCL memory space\n");
	memcpy(clbuffer_lGrdX, lGrdX.data, bufferSize_2D);
	memcpy(clbuffer_rGrdX, rGrdX.data, bufferSize_2D);

    int arg_num = 0;
    /* Setup the kernel arguments. */
    bool setKernelArgumentsSuccess = true;
	setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[CVC_LIMGR]));
	setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[CVC_LIMGG]));
	setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[CVC_LIMGB]));
	setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[CVC_RIMGR]));
	setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[CVC_RIMGG]));
	setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[CVC_RIMGB]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[CVC_LGRDX]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[CVC_RGRDX]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_int), &height));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_int), &width));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[CV_LCV]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[CV_RCV]));
    if (!setKernelArgumentsSuccess)
    {
		cleanUpOpenCL(NULL, NULL, NULL, NULL, NULL, 0);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    if(OCL_STATS) printf("CVC_cl: Running CVC Kernels\n");
    /* Enqueue the kernel */
    if (!checkSuccess(clEnqueueNDRangeKernel(*commandQueue, kernel, 3, NULL, globalWorksize, NULL, 0, NULL, &event)))
    {
        cleanUpOpenCL(NULL, NULL, NULL, NULL, NULL, 0);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* Wait for completion */
    if (!checkSuccess(clFinish(*commandQueue)))
    {
        cleanUpOpenCL(NULL, NULL, NULL, NULL, NULL, 0);
        cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* Print the profiling information for the event. */
    if(OCL_STATS) printProfilingInfo(event);
    /* Release the event object. */
    if (!checkSuccess(clReleaseEvent(event)))
    {
        cleanUpOpenCL(*context, *commandQueue, program, NULL, NULL, 0);
        cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    return 0;
}
