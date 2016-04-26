/*---------------------------------------------------------------------------
   BoxFilter.cpp - OpenCL Boxfilter Code
  ---------------------------------------------------------------------------
   Editor: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "BoxFilter.h"

BoxFilter::BoxFilter(int _height, int _width, int _radius, int _dispRange) :
	height((cl_int)_height), width((cl_int)_width), radius((cl_int)_radius), dispRange((cl_int)_dispRange)
{
    //fprintf(stderr, "OpenCL Guided Image Filtering method for Cost Computation\n" );

    //OpenCL Setup
    context = 0;
    commandQueue = 0;
    program = 0;
    kernel = 0;
    device = 0;
    numberOfMemoryObjects = 2;


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

    if (!createProgram(context, device, "assets/boxfilter_vector16.cl", &program))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
    }
	kernel = clCreateKernel(program, "boxfilter", &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
    }

	bufferSize = width * height * sizeof(cl_float) * dispRange;

    /* Create input buffers for the left and right image and gradient data, and an output buffer for the cost data. */
    bool createMemoryObjectsSuccess = true;
    memoryObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize, NULL, &errorNumber); //InData
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    memoryObjects[1] = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize, NULL, &errorNumber); //OutData
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
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_int), &width));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_int), &height));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[1]));
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    /* An event to associate with the Kernel. Allows us to retreive profiling information later. */
    event = 0;

    /* [Kernel size] */
    globalWorksize[0] = (size_t)width/16;
    //globalWorksize[0] = (size_t)width/8;
    globalWorksize[1] = (size_t)height;
    globalWorksize[2] = (size_t)dispRange;
}

BoxFilter::~BoxFilter(void)
{
	/* Release OpenCL objects. */
	cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
}

int BoxFilter::filter(Mat *in_ptr, Mat *out_ptr)
{
	//fprintf(stderr, "%s Boxfilter Kernel: GWS = %dx%dx%d\n", (dispRange > 1) ? "3D" : "2D", (int)globalWorksize[0], (int)globalWorksize[1], (int)globalWorksize[2]);
	for(int d = 0; d < dispRange; d++)
	{
		/* Map the input image memory objects to host side pointers. */
		clEnqueueWriteBuffer(commandQueue, memoryObjects[0], CL_TRUE, d*bufferSize/dispRange, bufferSize/dispRange, in_ptr[d].data, 0, NULL, NULL);
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

	//cout << "BoxFilter ";
	//printProfilingInfo(event);
//    /* Release the event object. */
//    if (!checkSuccess(clReleaseEvent(event)))
//    {
//        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
//        cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
//        return 1;
//    }

	for(int d = 0; d < dispRange; d++)
	{
		/* Map the input image memory objects to host side pointers. */
		clEnqueueReadBuffer(commandQueue, memoryObjects[1], CL_TRUE, d*bufferSize/dispRange, bufferSize/dispRange, out_ptr[d].data, 0, NULL, NULL);
	}
    return 0;
}
