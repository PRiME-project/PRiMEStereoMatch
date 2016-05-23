/*---------------------------------------------------------------------------
   CVC_cl_image.cpp - OpenCL Cost Volume Construction Code
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "CVC_cl_image.h"

CVC_cli::CVC_cli(Mat l, const int d) : lImg_ref(l), maxDis(d)
{
    //fprintf(stderr, "OpenCL Colours and Gradients method for Cost Computation\n" );

    //OpenCL Setup
    context = 0;
    commandQueue = 0;
    program = 0;
    kernel = 0;
    device = 0;
    numberOfMemoryObjects = 6;

    if (!createContext(&context))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create an OpenCL context. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    if (!createCommandQueue(context, &commandQueue, &device))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create the OpenCL command queue. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    if (!createProgram(context, device, "assets/cvc_image.cl", &program))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
    }

    kernel = clCreateKernel(program, "cvc_image", &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    width = (cl_int)lImg_ref.cols;
    height = (cl_int)lImg_ref.rows;
    dispRange = (cl_int)maxDis;

    format_color.image_channel_data_type = CL_FLOAT;
    format_color.image_channel_order = CL_RGBA;
    format_grad.image_channel_data_type = CL_FLOAT;
    format_grad.image_channel_order = CL_LUMINANCE;
    format_cv.image_channel_data_type = CL_FLOAT;
    format_cv.image_channel_order = CL_LUMINANCE;

    /* Create input buffers for the left and right image and gradient data, and an output buffer for the cost data. */
    bool createMemoryObjectsSuccess = true;
    memoryObjects[0] = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, &format_color, width, height, 0, NULL, &errorNumber); //lImgData
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    memoryObjects[1] = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, &format_color, width, height, 0, NULL, &errorNumber); //rImgData
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    memoryObjects[2] = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, &format_grad, width, height, 0, NULL, &errorNumber); //lGrdX
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    memoryObjects[3] = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, &format_grad, width, height, 0, NULL, &errorNumber); //rGrdX
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    memoryObjects[4] = clCreateImage3D(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, &format_cv, width, height, maxDis, 0, 0, NULL, &errorNumber);  //lcostVol
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    memoryObjects[5] = clCreateImage3D(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, &format_cv, width, height, maxDis, 0, 0, NULL, &errorNumber);  //rcostVol
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    if (!createMemoryObjectsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL buffers. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    int arg_num = 0;
    /* Setup the kernel arguments. */
    bool setKernelArgumentsSuccess = true;
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[1]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[2]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[3]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_int), &height));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_int), &width));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[4]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[5]));
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    /* An event to associate with the Kernel. Allows us to retreive profiling information later. */
    event = 0;

    /* [Kernel size] */
    /*
     * Each instance of the kernel operates on a single pixel portion of the image.
     * Therefore, the global work size is the number of pixel.
     */
    //globalWorksize[0] = (size_t)width;
    globalWorksize[0] = (size_t)width;
    globalWorksize[1] = (size_t)height;
    globalWorksize[2] = (size_t)dispRange;
    /* [Kernel size] */
}
CVC_cli::~CVC_cli(void)
{
    /* Release OpenCL objects. */
    cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
}

int CVC_cli::buildCV(const Mat& lImg, const Mat& rImg, Mat* lcostVol, Mat* rcostVol)
{
    //CV_Assert( lImg.type() == CV_32FC3 && rImg.type() == CV_32FC3 );

	Mat lImg4 = Mat::zeros(height, width, CV_32FC4);
	Mat rImg4 = Mat::zeros(height, width, CV_32FC4);

	cvtColor(lImg, lImg4, CV_RGB2RGBA );
	cvtColor(rImg, rImg4, CV_RGB2RGBA );

	cvtColor(lImg, lGray, CV_RGB2GRAY );
	cvtColor(rImg, rGray, CV_RGB2GRAY );

    //Sobel filter to compute X gradient     <-- investigate Mali Sobel OpenCL kernel
    Sobel( lGray, lGrdX, CV_32F, 1, 0, 1 ); // ex time 16 -17ms
	Sobel( rGray, rGrdX, CV_32F, 1, 0, 1 ); // for both
	lGrdX += 0.5;
	rGrdX += 0.5;
	origin[2] = 0;

	/* Map the input image memory objects to host side pointers. */
	bool EnqueueWriteBufferSuccess = true;
	EnqueueWriteBufferSuccess &= checkSuccess(clEnqueueWriteImage(commandQueue, memoryObjects[0], CL_TRUE, origin, region2D, 0, 0, lImg4.data, 0, NULL, NULL));
	EnqueueWriteBufferSuccess &= checkSuccess(clEnqueueWriteImage(commandQueue, memoryObjects[1], CL_TRUE, origin, region2D, 0, 0, rImg4.data, 0, NULL, NULL));
	EnqueueWriteBufferSuccess &= checkSuccess(clEnqueueWriteImage(commandQueue, memoryObjects[2], CL_TRUE, origin, region2D, 0, 0, lGrdX.data, 0, NULL, NULL));
	EnqueueWriteBufferSuccess &= checkSuccess(clEnqueueWriteImage(commandQueue, memoryObjects[3], CL_TRUE, origin, region2D, 0, 0, rGrdX.data, 0, NULL, NULL));
	if (!EnqueueWriteBufferSuccess)
	{
	   cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
	   cerr << "Mapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
	}

    /* Enqueue the kernel */
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernel, 3, NULL, globalWorksize, NULL, 0, NULL, &event)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* Wait for completion */
    if (!checkSuccess(clFinish(commandQueue)))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

	//cout << "CVC ";
    /* Print the profiling information for the event. */
    //printProfilingInfo(event);
    /* Release the event object. */
//    if (!checkSuccess(clReleaseEvent(event)))
//    {
//        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
//        cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
//        return 1;
//    }

    for( int d = 0; d < maxDis; d ++ )
    {
		origin[2] = d;
		bool EnqueueReadBufferSuccess = true;
		EnqueueReadBufferSuccess &= checkSuccess(clEnqueueReadImage(commandQueue, memoryObjects[4], CL_TRUE, origin, region2D, 0, 0, lcostVol[d].data, 0, NULL, NULL));
		EnqueueReadBufferSuccess &= checkSuccess(clEnqueueReadImage(commandQueue, memoryObjects[5], CL_TRUE, origin, region2D, 0, 0, rcostVol[d].data, 0, NULL, NULL));
		if (!EnqueueReadBufferSuccess)
		{
			cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
			cerr << "Mapping memory objects failed. " << __FILE__ << ":"<< __LINE__ << endl;
			return 1;
		}
    }
    return 0;
}
