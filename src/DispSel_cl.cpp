/*---------------------------------------------------------------------------
   DispSel_cl.cpp - OpenCL Disparity Selection Code
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "DispSel_cl.h"

DispSel_cl::DispSel_cl(cl_context* context, cl_command_queue* commandQueue, cl_device_id device,
						Mat* I, const int d) : maxDis(d), context(context), commandQueue(commandQueue)
{
    //fprintf(stderr, "Winner-Takes-All Disparity Selection\n" );

    //OpenCL Setup
    program = 0;
//    imgType = I->type() & CV_MAT_DEPTH_MASK;

    if (!createProgram(*context, device, FILE_DS_PROG, &program))
    {
        cleanUpOpenCL(NULL, NULL, NULL, kernel, NULL, 0);
        cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
    }

//	if(imgType == CV_32F)
//	{
		strcpy(kernel_name, "dispsel_float");
		//strcpy(kernel_name, "dispsel_double");
//	}
//	else if(imgType == CV_8U)
//	{
//		strcpy(kernel_name, "dispsel_uchar");
//	}
//    else{
//		printf("DS_cl: Error - Unrecognised data type in processing! (DS_cl)\n");
//		exit(1);
//    }
	kernel = clCreateKernel(program, kernel_name, &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        cleanUpOpenCL(NULL, NULL, NULL, kernel, NULL, 0);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
    }
    else{
		printf("DispSel_cl: OpenCL kernels created.\n");
	}

    width = (cl_int)I->cols;
    height = (cl_int)I->rows;

	//OpenCL Buffers in accending size order
	//bufferSize_2D_8UC1 = width * height * sizeof(cl_float);
	bufferSize_2D_8UC1 = width * height * sizeof(cl_char);

    /* An event to associate with the Kernel. Allows us to retreive profiling information later. */
    event = 0;

    //Kernel size
    globalWorksize[0] = (size_t)width;
    globalWorksize[1] = (size_t)height;
}
DispSel_cl::~DispSel_cl(void)
{
    /* Release OpenCL objects. */
	cleanUpOpenCL(NULL, NULL, NULL, kernel, NULL, 0);
}

int DispSel_cl::CVSelect(cl_mem *memoryObjects, Mat& ldispMap, Mat& rdispMap)
{

	namedWindow("dispPreview", CV_WINDOW_AUTOSIZE);
	imshow("dispPreview", ldispMap);
	waitKey(0);

	int arg_num = 0;
    /* Setup the kernel arguments. */
    bool setKernelArgumentsSuccess = true;
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[CV_LCV]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[CV_RCV]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_int), &height));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_int), &width));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_int), &maxDis));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[DS_LDM]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[DS_RDM]));
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(NULL, NULL, NULL, kernel, NULL, 0);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    if(OCL_STATS) printf("DS_cl: Running DispSel Kernels\n");
    /* Enqueue the kernel */
    if (!checkSuccess(clEnqueueNDRangeKernel(*commandQueue, kernel, 2, NULL, globalWorksize, NULL, 0, NULL, &event)))
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
	imshow("dispPreview", ldispMap);
	waitKey(0);

	/* Map the output memory objects to host side pointers. */
	bool EnqueueMapBufferSuccess = true;
	cl_char *clbuffer_lDispMap = (cl_char*)clEnqueueMapBuffer(*commandQueue, memoryObjects[DS_LDM], CL_TRUE, CL_MAP_READ, 0, bufferSize_2D_8UC1, 0, NULL, NULL, &errorNumber);
	EnqueueMapBufferSuccess &= checkSuccess(errorNumber);
	cl_char *clbuffer_rDispMap = (cl_char*)clEnqueueMapBuffer(*commandQueue, memoryObjects[DS_RDM], CL_TRUE, CL_MAP_READ, 0, bufferSize_2D_8UC1, 0, NULL, NULL, &errorNumber);
	EnqueueMapBufferSuccess &= checkSuccess(errorNumber);
	if (!EnqueueMapBufferSuccess)
	{
	   cleanUpOpenCL(*context, *commandQueue, program, kernel, NULL, 0);
	   cerr << "Mapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
	}

	memcpy(ldispMap.data, clbuffer_lDispMap, bufferSize_2D_8UC1);
	memcpy(rdispMap.data, clbuffer_rDispMap, bufferSize_2D_8UC1);

	imshow("dispPreview", ldispMap);
	waitKey(0);

//    return 0;
}
