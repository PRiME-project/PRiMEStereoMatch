/*---------------------------------------------------------------------------
   DispSel_cl.cpp - OpenCL Disparity Selection Code
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "DispSel_cl.h"

DispSel_cl::DispSel_cl(cl_context* context, cl_command_queue* commandQueue, cl_device_id device,
						Mat l, const int d) : lImg_ref(l), maxDis(d), context(context), commandQueue(commandQueue)
{
    //fprintf(stderr, "Winner-Takes-All Disparity Selection\n" );

    //OpenCL Setup
    program = 0;

    if (!createProgram(*context, device, "assets/dispsel_float.cl", &program))
    //if (!createProgram(context, device, "assets/dispsel_double.cl", &program))
    {
        cleanUpOpenCL(NULL, NULL, NULL, kernel, NULL, 0);
        cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
    }

    kernel = clCreateKernel(program, "dispsel", &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        cleanUpOpenCL(NULL, NULL, NULL, kernel, NULL, 0);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    width = (cl_int)lImg_ref.cols;
    height = (cl_int)lImg_ref.rows;

	//Buffers in accending size order
	bufferSize_2D_C1C = width * height * sizeof(cl_char);
	bufferSize_2D_C1F = width * height * sizeof(cl_float);
	bufferSize_3D_C1F = width * height * maxDis * sizeof(cl_float);

    /* An event to associate with the Kernel. Allows us to retreive profiling information later. */
    event = 0;

    /* [Kernel size] */
    /*
     * Each instance of the kernel operates on a single pixel portion of the image.
     * Therefore, the global work size is the number of pixel.
     */
    globalWorksize[0] = (size_t)width;
    globalWorksize[1] = (size_t)height;
    /* [Kernel size] */
}
DispSel_cl::~DispSel_cl(void)
{
    /* Release OpenCL objects. */
	cleanUpOpenCL(NULL, NULL, NULL, kernel, NULL, 0);
}

int DispSel_cl::CVSelect(cl_mem *memoryObjects, Mat& ldispMap, Mat& rdispMap)
{
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

	/* Map the input memory objects to host side pointers. */
	bool EnqueueMapBufferSuccess = true;
	cl_char *clbuffer_lDispMap = (cl_char*)clEnqueueMapBuffer(*commandQueue, memoryObjects[DS_LDM], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, bufferSize_2D_C1C, 0, NULL, NULL, &errorNumber);
	EnqueueMapBufferSuccess &= checkSuccess(errorNumber);
	cl_char *clbuffer_rDispMap = (cl_char*)clEnqueueMapBuffer(*commandQueue, memoryObjects[DS_RDM], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, bufferSize_2D_C1C, 0, NULL, NULL, &errorNumber);
	EnqueueMapBufferSuccess &= checkSuccess(errorNumber);
	if (!EnqueueMapBufferSuccess)
	{
	   cleanUpOpenCL(*context, *commandQueue, program, kernel, NULL, 0);
	   cerr << "Mapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
	}

	memcpy(ldispMap.data, clbuffer_lDispMap, bufferSize_2D_C1C);
	memcpy(rdispMap.data, clbuffer_rDispMap, bufferSize_2D_C1C);
    return 0;
}
