/*---------------------------------------------------------------------------
   CVC_cl.cpp - OpenCL Cost Volume Construction Code
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "CVC_cl.h"

CVC_cl::CVC_cl(Mat l, const int d) : lImg_ref(l), maxDis(d)
{
    //fprintf(stderr, "OpenCL Colours and Gradients method for Cost Computation\n" );

    //OpenCL Setup
    context = 0;
    commandQueue = 0;
    program = 0;
    kernel = 0;
    device = 0;
    numberOfMemoryObjects = 6;
    for(int m = 0; m < numberOfMemoryObjects; m++)
		memoryObjects[m] = 0;

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

    //if (!createProgram(context, device, "assets/cvc_novector.cl", &program))
    //if (!createProgram(context, device, "assets/cvc_double.cl", &program))
    //if (!createProgram(context, device, "assets/cvc_float.cl", &program))
    if (!createProgram(context, device, "assets/cvc_float.cl", &program))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
    }

    kernel = clCreateKernel(program, "cvc", &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    width = (cl_int)lImg_ref.cols;
    height = (cl_int)lImg_ref.rows;
    channels = (cl_int)lImg_ref.channels();
    dispRange = (cl_int)maxDis;

    bufferSize_color = width * height * sizeof(cl_float) * channels;
    bufferSize_grad = width * height * sizeof(cl_float);
    bufferSize_costVol = width * height * maxDis * sizeof(cl_float);

    /* Create input buffers for the left and right image and gradient data, and an output buffer for the cost data. */
    bool createMemoryObjectsSuccess = true;
    memoryObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize_color, NULL, &errorNumber); //lImgData
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    memoryObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize_color, NULL, &errorNumber); //rImgData
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    memoryObjects[2] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize_grad, NULL, &errorNumber); //lGrdX
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    memoryObjects[3] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize_grad, NULL, &errorNumber); //rGrdX
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    memoryObjects[4] = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize_costVol, NULL, &errorNumber);  //costVol
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    memoryObjects[5] = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize_costVol, NULL, &errorNumber);  //costVol
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
    globalWorksize[0] = (size_t)width/4;
    globalWorksize[1] = (size_t)height;
    globalWorksize[2] = (size_t)dispRange;
    /* [Kernel size] */
}
CVC_cl::~CVC_cl(void)
{
    /* Release OpenCL objects. */
    cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
}

int CVC_cl::buildCV(const Mat& lImg, const Mat& rImg, Mat* lcostVol, Mat* rcostVol)
{
    CV_Assert( lImg.type() == CV_32FC3 && rImg.type() == CV_32FC3 );

	//lImg.convertTo( tmp, CV_32F );
	cvtColor(lImg, lGray, CV_RGB2GRAY );
	//rImg.convertTo( tmp, CV_32F );
	cvtColor(rImg, rGray, CV_RGB2GRAY );

    //Sobel filter to compute X gradient     <-- investigate Mali Sobel OpenCL kernel
    Sobel( lGray, lGrdX, CV_32F, 1, 0, 1 ); // ex time 16 -17ms
	Sobel( rGray, rGrdX, CV_32F, 1, 0, 1 ); // for both
	lGrdX += 0.5;
	rGrdX += 0.5;

	/* Map the input image memory objects to host side pointers. */
	bool EnqueueWriteBufferSuccess = true;
	EnqueueWriteBufferSuccess &= checkSuccess(clEnqueueWriteBuffer(commandQueue, memoryObjects[0], CL_TRUE, 0, bufferSize_color, lImg.data, 0, NULL, NULL));
	EnqueueWriteBufferSuccess &= checkSuccess(clEnqueueWriteBuffer(commandQueue, memoryObjects[1], CL_TRUE, 0, bufferSize_color, rImg.data, 0, NULL, NULL));
	EnqueueWriteBufferSuccess &= checkSuccess(clEnqueueWriteBuffer(commandQueue, memoryObjects[2], CL_TRUE, 0, bufferSize_grad, lGrdX.data, 0, NULL, NULL));
	EnqueueWriteBufferSuccess &= checkSuccess(clEnqueueWriteBuffer(commandQueue, memoryObjects[3], CL_TRUE, 0, bufferSize_grad, rGrdX.data, 0, NULL, NULL));
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
		bool EnqueueReadBufferSuccess = true;
		EnqueueReadBufferSuccess &= checkSuccess(clEnqueueReadBuffer(commandQueue, memoryObjects[4], CL_TRUE, d*bufferSize_grad, bufferSize_grad, lcostVol[d].data, 0, NULL, NULL));
		EnqueueReadBufferSuccess &= checkSuccess(clEnqueueReadBuffer(commandQueue, memoryObjects[5], CL_TRUE, d*bufferSize_grad, bufferSize_grad, rcostVol[d].data, 0, NULL, NULL));
		if (!EnqueueReadBufferSuccess)
		{
			cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
			cerr << "Mapping memory objects failed. " << __FILE__ << ":"<< __LINE__ << endl;
			return 1;
		}
    }
    return 0;
}
