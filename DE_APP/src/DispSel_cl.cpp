/*---------------------------------------------------------------------------
   DispSel_cl.cpp - OpenCL Disparity Selection Code
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "DispSel_cl.h"

DispSel_cl::DispSel_cl(Mat l, const int d) : lImg_ref(l), maxDis(d)
{
    //fprintf(stderr, "Winner-Takes-All Disparity Selection\n" );

    //OpenCL Setup
    context = 0;
    commandQueue = 0;
    program = 0;
    kernel = 0;
    device = 0;
    numberOfMemoryObjects = 4;

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

    if (!createProgram(context, device, "assets/dispsel.cl", &program))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
    }

    kernel = clCreateKernel(program, "dispsel", &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    width = (cl_int)lImg_ref.cols;
    height = (cl_int)lImg_ref.rows;

	bufferSize_char = width * height * sizeof(cl_char);
    bufferSize_grad = width * height * sizeof(cl_float);
    bufferSize_costVol = width * height * maxDis * sizeof(cl_float);

    /* Create input buffers for the left and right costVol and output buffers for the disparity maps.*/
    bool createMemoryObjectsSuccess = true;
    memoryObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize_costVol, NULL, &errorNumber); //lcostVol
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    memoryObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize_costVol, NULL, &errorNumber); //rcostVol
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    memoryObjects[2] = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize_char, NULL, &errorNumber); //lDispMap
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    memoryObjects[3] = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize_char, NULL, &errorNumber); //rDispMap
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);
    if (!createMemoryObjectsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed to create OpenCL buffers. " << __FILE__ << ":"<< __LINE__ << endl;
    }

//    /* Map the input image memory objects to host side pointers. */
    bool EnqueueMapBufferSuccess = true;
    lcostVol_cl = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[0], CL_TRUE, CL_MAP_WRITE, 0, bufferSize_costVol, 0, NULL, NULL, &errorNumber);
    EnqueueMapBufferSuccess &= checkSuccess(errorNumber);
    rcostVol_cl = (cl_float*)clEnqueueMapBuffer(commandQueue, memoryObjects[1], CL_TRUE, CL_MAP_WRITE, 0, bufferSize_costVol, 0, NULL, NULL, &errorNumber);
    EnqueueMapBufferSuccess &= checkSuccess(errorNumber);
    if (!EnqueueMapBufferSuccess)
    {
       cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
       cerr << "Mapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
    }

    int arg_num = 0;
    /* Setup the kernel arguments. */
    bool setKernelArgumentsSuccess = true;
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[1]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_int), &height));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_int), &width));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_int), &maxDis));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[2]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &memoryObjects[3]));
    if (!setKernelArgumentsSuccess)
    {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    /* An event to associate with the Kernel. Allows us to retreive profiling information later. */
    //event = 0;

    /* [Kernel size] */
    /*
     * Each instance of the kernel operates on a single pixel portion of the image.
     * Therefore, the global work size is the number of pixel.
     */
    globalWorksize[0] = (size_t)width;
    globalWorksize[1] = (size_t)height;
    /* [Kernel size] */

    /* Map the arrays holding the output cost volume. */
    bool mapMemoryObjectsSuccess = true;
    ldispMap_cl = (cl_char*)clEnqueueMapBuffer(commandQueue, memoryObjects[2], CL_TRUE, CL_MAP_READ, 0, bufferSize_char, 0, NULL, NULL, &errorNumber);
    mapMemoryObjectsSuccess &= checkSuccess(errorNumber);
    rdispMap_cl = (cl_char*)clEnqueueMapBuffer(commandQueue, memoryObjects[3], CL_TRUE, CL_MAP_READ, 0, bufferSize_char, 0, NULL, NULL, &errorNumber);
    mapMemoryObjectsSuccess &= checkSuccess(errorNumber);
    if (!mapMemoryObjectsSuccess)
    {
       cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
       cerr << "Mapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
    }
}
DispSel_cl::~DispSel_cl(void)
{
	/* Unmap the memory. */
    bool unmapMemoryObjectsSuccess = true;
    unmapMemoryObjectsSuccess &= checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[2], ldispMap_cl, 0, NULL, NULL));
    unmapMemoryObjectsSuccess &= checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[3], rdispMap_cl, 0, NULL, NULL));
    if (!unmapMemoryObjectsSuccess)
    {
       cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
    }

    /* Release OpenCL objects. */
    cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
}

int DispSel_cl::CVSelect(Mat* lcostVol, Mat* rcostVol, Mat& ldispMap, Mat& rdispMap)
{
    for(int d = 0; d < maxDis; d ++ )
    {
        Mat& lcost = lcostVol[d];
        Mat& rcost = rcostVol[d];
        memcpy(lcostVol_cl+d*height*width, lcost.data, bufferSize_grad);
        memcpy(rcostVol_cl+d*height*width, rcost.data, bufferSize_grad);

		/* Map the input image memory objects to host side pointers. */
//		bool EnqueueWriteBufferSuccess = true;
//		EnqueueWriteBufferSuccess &= checkSuccess(clEnqueueWriteBuffer(commandQueue, memoryObjects[0], CL_TRUE, d*bufferSize_grad, bufferSize_grad, lcost.data, 0, NULL, NULL));
//		EnqueueWriteBufferSuccess &= checkSuccess(clEnqueueWriteBuffer(commandQueue, memoryObjects[1], CL_TRUE, d*bufferSize_grad, bufferSize_grad, rcost.data, 0, NULL, NULL));
//		if (!EnqueueWriteBufferSuccess)
//		{
//		   cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
//		   cerr << "Mapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
//		}
    }


    /* Enqueue the kernel */
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorksize, NULL, 0, NULL, NULL)))
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

    /* Print the profiling information for the event. */
    //printProfilingInfo(event);
    /* Release the event object. */
//    if (!checkSuccess(clReleaseEvent(event)))
//    {
//        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
//        cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
//        return 1;
//    }

//    bool EnqueueReadBufferSuccess = true;
//	EnqueueReadBufferSuccess &= checkSuccess(clEnqueueReadBuffer(commandQueue, memoryObjects[2], CL_TRUE, 0, bufferSize_char, ldispMap.data, 0, NULL, NULL));
//	EnqueueReadBufferSuccess &= checkSuccess(clEnqueueReadBuffer(commandQueue, memoryObjects[3], CL_TRUE, 0, bufferSize_char, rdispMap.data, 0, NULL, NULL));
//	if (!EnqueueReadBufferSuccess)
//    {
//        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numberOfMemoryObjects);
//        cerr << "Mapping memory objects failed. " << __FILE__ << ":"<< __LINE__ << endl;
//        return 1;
//    }
	memcpy(ldispMap.data, ldispMap_cl, bufferSize_char);
	memcpy(rdispMap.data, rdispMap_cl, bufferSize_char);
    return 0;
}
