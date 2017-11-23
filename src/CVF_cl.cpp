/*---------------------------------------------------------------------------
   CVF_cl.cpp - OpenCL Cost Volume Filter Code
              - Guided Image Filter
              - Box Filter
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "CVF_cl.h"

float get_rt_cvf_cl(){
	struct timespec realtime;
	clock_gettime(CLOCK_MONOTONIC,&realtime);
	return (float)(realtime.tv_sec*1000000+realtime.tv_nsec/1000);
}

CVF_cl::CVF_cl(cl_context* context, cl_command_queue* commandQueue, cl_device_id device, Mat* I, const int d) :
				context(context), commandQueue(commandQueue), maxDis(d)
{
	//OpenCL Setup
    program = 0;
//    imgType = I->type() & CV_MAT_DEPTH_MASK;

    if (!createProgram(*context, device, FILE_CVF_PROG, &program))
    {
        cleanUpOpenCL(*context, *commandQueue, program, kernel_split, NULL, 0);
        cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
    }

//	if(imgType == CV_32F)
//	{
		kernel_mmsd = clCreateKernel(program, "EWMul_SameDim_32F", &errorNumber);
		kernel_mmdd = clCreateKernel(program, "EWMul_DiffDim_32F", &errorNumber);
		kernel_mdsd = clCreateKernel(program, "EWDiv_SameDim_32F", &errorNumber);
		kernel_split = clCreateKernel(program, "Split_32F", &errorNumber);
		kernel_sub = clCreateKernel(program, "Subtract_32F", &errorNumber);
		kernel_add = clCreateKernel(program, "Add_32F", &errorNumber);
		kernel_centf = clCreateKernel(program, "cent_filter_32F", &errorNumber);
		kernel_var = clCreateKernel(program, "var_math_32F", &errorNumber);
		kernel_bf = clCreateKernel(program, "BoxFilter_32F", &errorNumber);
//		printf("CVF_cl: Float (_32F) OpenCL kernel versions created in context.\n");
//	}
//	else if(imgType == CV_8U)
//	{
//		kernel_mmsd = clCreateKernel(program, "EWMul_SameDim_8U", &errorNumber);
//		kernel_mmdd = clCreateKernel(program, "EWMul_DiffDim_8U", &errorNumber);
//		kernel_mdsd = clCreateKernel(program, "EWDiv_SameDim_8U", &errorNumber);
//		kernel_split = clCreateKernel(program, "Split_8U", &errorNumber);
//		kernel_sub = clCreateKernel(program, "Subtract_8U", &errorNumber);
//		kernel_add = clCreateKernel(program, "Add_8U", &errorNumber);
//		kernel_centf = clCreateKernel(program, "cent_filter_8U", &errorNumber);
//		kernel_var = clCreateKernel(program, "var_math_8U", &errorNumber);
//		kernel_bfc_rows = clCreateKernel(program, "BoxRows_8U", &errorNumber);
//		kernel_bfc_cols = clCreateKernel(program, "BoxCols_8U", &errorNumber);
//		printf("CVF_cl: Char (_8U) OpenCL kernel versions created in context.\n");
//	}
//    else{
//		printf("CVF_cl: Error - Unrecognised data type in processing! (CVF_cl)\n");
//		exit(1);
//    }
    if (!checkSuccess(errorNumber))
    {
        cleanUpOpenCL(*context, *commandQueue, program, NULL, NULL, 0);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
		exit(1);
    }
    else{
		printf("CVF_cl: OpenCL kernel versions created in context.\n");
    }

    /* An event to associate with the Kernel. Allows us to retreive profiling information later. */
    event = 0;

	width = I->cols;
	height = I->rows;

	//OpenCL Buffers that are type dependent (in accending size order)
//	if(imgType == CV_32F)
//	{
		bufferSize_2D = width * height * sizeof(cl_float);
		bufferSize_3D = width * height * maxDis * sizeof(cl_float);
//	}
//	else if(imgType == CV_8U)
//	{
//		bufferSize_2D = width * height * sizeof(cl_uchar);
//		bufferSize_3D = width * height * maxDis * sizeof(cl_uchar);
//	}

    globalWorksize_3D[0] = (size_t)width;
    globalWorksize_3D[1] = (size_t)height;
    globalWorksize_3D[2] = (size_t)maxDis;

    globalWorksize_2D[0] = (size_t)width;
    globalWorksize_2D[1] = (size_t)height;
    globalWorksize_2D[2] = (size_t)1;

    globalWorksize_split[0] = (size_t)width/3;
    globalWorksize_split[1] = (size_t)height;

    globalWorksize_bf_3D[0] = (size_t)width/16;
    globalWorksize_bf_3D[1] = (size_t)height;
    globalWorksize_bf_3D[2] = (size_t)maxDis;

    globalWorksize_bf_2D[0] = (size_t)width/16;
    globalWorksize_bf_2D[1] = (size_t)height;
    globalWorksize_bf_2D[2] = (size_t)1;

    globalWorksize_bfc_2D[0] = (size_t)height;
    globalWorksize_bfc_2D[1] = (size_t)1;
    globalWorksize_bfc_2D[2] = (size_t)1;

    globalWorksize_bfc_3D[0] = (size_t)height;
    globalWorksize_bfc_3D[1] = (size_t)1;
    globalWorksize_bfc_3D[2] = (size_t)maxDis;

	bool createMemoryObjectsSuccess = true;

	Ixx = new cl_mem[6];
	mean_I = new cl_mem[3]; //r, g, b
	mean_Ixx = new cl_mem[6]; //rr, rg, rb, gg, gb, bb
	var_I = new cl_mem[6]; //rr, rg, rb, gg, gb, bb
	cov_Ip = new cl_mem[3];
	a = new cl_mem[3];
	for(int i = 0; i < 6; i++)
	{
		Ixx[i] = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D, NULL, &errorNumber);
		createMemoryObjectsSuccess &= checkSuccess(errorNumber);
		mean_Ixx[i] = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D, NULL, &errorNumber);
		createMemoryObjectsSuccess &= checkSuccess(errorNumber);
		var_I[i] = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D, NULL, &errorNumber);
		createMemoryObjectsSuccess &= checkSuccess(errorNumber);
		if(i<3)
		{
			mean_I[i] = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D, NULL, &errorNumber);
			createMemoryObjectsSuccess &= checkSuccess(errorNumber);
			cov_Ip[i] = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D, NULL, &errorNumber);
			createMemoryObjectsSuccess &= checkSuccess(errorNumber);
			a[i] = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D, NULL, &errorNumber);
			createMemoryObjectsSuccess &= checkSuccess(errorNumber);
		}
	}

	mean_cv = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);

	tmp_3DA_r = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	tmp_3DA_g = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	tmp_3DA_b = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	tmp_3DB_r = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	tmp_3DB_g = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	tmp_3DB_b = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);

//	bf2Dtmp = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D, NULL, &errorNumber);
//	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
//	bf3Dtmp = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D, NULL, &errorNumber);
//	createMemoryObjectsSuccess &= checkSuccess(errorNumber);

	if (!createMemoryObjectsSuccess)
	{
		cerr << "Failed to create OpenCL buffers. " << __FILE__ << ":"<< __LINE__ << endl;
	}
    printf("Allocated OpenCL Buffers\n");
}

CVF_cl::~CVF_cl(void)
{
	for(int i = 0; i < 6; i++)
	{
		clReleaseMemObject(var_I[i]);
		clReleaseMemObject(mean_Ixx[i]);
		if(i<3)
		{
			clReleaseMemObject(mean_I[i]);
			clReleaseMemObject(a[i]);
			clReleaseMemObject(cov_Ip[i]);
		}
	}

	clReleaseMemObject(mean_cv);
	clReleaseMemObject(tmp_3DA_r);
	clReleaseMemObject(tmp_3DA_g);
	clReleaseMemObject(tmp_3DA_b);
	clReleaseMemObject(tmp_3DB_r);
	clReleaseMemObject(tmp_3DB_g);
	clReleaseMemObject(tmp_3DB_b);
//	clReleaseMemObject(bf2Dtmp);
//	clReleaseMemObject(bf3Dtmp);
}

int CVF_cl::preprocess(cl_mem* ImgR, cl_mem* ImgG, cl_mem* ImgB)
{
    Ir = ImgR;
    Ig = ImgG;
    Ib = ImgB;

    //mean_I
//	if(imgType == CV_32F)
//	{
		boxfilter(Ir, &mean_I[0], globalWorksize_bf_2D);
		boxfilter(Ig, &mean_I[1], globalWorksize_bf_2D);
		boxfilter(Ib, &mean_I[2], globalWorksize_bf_2D);
//	}
//	else if(imgType == CV_8U)
//	{
//		boxfilter(Ir, &bf2Dtmp, &mean_I[0], globalWorksize_bfc_2D);
//		boxfilter(Ig, &bf2Dtmp, &mean_I[1], globalWorksize_bfc_2D);
//		boxfilter(Ib, &bf2Dtmp, &mean_I[2], globalWorksize_bfc_2D);
//	}

	elementwiseMulSD(Ir, Ir, &Ixx[0], globalWorksize_2D);
	elementwiseMulSD(Ir, Ig, &Ixx[1], globalWorksize_2D);
	elementwiseMulSD(Ir, Ib, &Ixx[2], globalWorksize_2D);
	elementwiseMulSD(Ig, Ig, &Ixx[3], globalWorksize_2D);
	elementwiseMulSD(Ig, Ib, &Ixx[4], globalWorksize_2D);
	elementwiseMulSD(Ib, Ib, &Ixx[5], globalWorksize_2D);

	for(int i = 0; i < 6; i++)
	{
//		if(imgType == CV_32F)
			boxfilter(&Ixx[i], &mean_Ixx[i], globalWorksize_bf_2D);
//		else if(imgType == CV_8U)
//			boxfilter(&Ixx[i], &bf2Dtmp, &mean_Ixx[i], globalWorksize_bfc_2D);
	}

	preproc_maths(mean_I, mean_Ixx, var_I, globalWorksize_2D);

	return 0;
}

int CVF_cl::filterCV(cl_mem* cl_costVol)
{
//	if(imgType == CV_32F)
		boxfilter(cl_costVol, &mean_cv, globalWorksize_bf_3D);
//	else if(imgType == CV_8U)
//		boxfilter(cl_costVol, &bf3Dtmp, &mean_cv, globalWorksize_bfc_3D);

	elementwiseMulDD(Ir, cl_costVol, &tmp_3DA_r); //Icv_r
	elementwiseMulDD(Ig, cl_costVol, &tmp_3DA_g); //Icv_g
	elementwiseMulDD(Ib, cl_costVol, &tmp_3DA_b); //Icv_b

//	if(imgType == CV_32F){
		boxfilter(&tmp_3DA_r, &tmp_3DB_r, globalWorksize_bf_3D); //mean_Icv_r
		boxfilter(&tmp_3DA_g, &tmp_3DB_g, globalWorksize_bf_3D); //mean_Icv_g
		boxfilter(&tmp_3DA_b, &tmp_3DB_b, globalWorksize_bf_3D); //mean_Icv_b
//	} else if(imgType == CV_8U){
//		boxfilter(&tmp_3DA_r, &bf3Dtmp, &tmp_3DB_r, globalWorksize_bfc_3D); //mean_Icv_r
//		boxfilter(&tmp_3DA_g, &bf3Dtmp, &tmp_3DB_g, globalWorksize_bfc_3D); //mean_Icv_g
//		boxfilter(&tmp_3DA_b, &bf3Dtmp, &tmp_3DB_b, globalWorksize_bfc_3D); //mean_Icv_b
//	}

	elementwiseMulDD(&mean_I[0], &mean_cv, &tmp_3DA_r); //mean_Ir_cv
	elementwiseMulDD(&mean_I[1], &mean_cv, &tmp_3DA_g); //mean_Ig_cv
	elementwiseMulDD(&mean_I[2], &mean_cv, &tmp_3DA_b); //mean_Ib_cv

	sub(&tmp_3DB_r, &tmp_3DA_r, &cov_Ip[0], globalWorksize_3D); //cov_Icv_r = mean_Icv_r - mean_Ir_cv
	sub(&tmp_3DB_g, &tmp_3DA_g, &cov_Ip[1], globalWorksize_3D); //cov_Icv_g = mean_Icv_g - mean_Ig_cv
	sub(&tmp_3DB_b, &tmp_3DA_b, &cov_Ip[2], globalWorksize_3D); //cov_Icv_b = mean_Icv_b - mean_Ib_cv

    // variance of I in each local patch: the matrix Sigma in Eqn (14).
    // Note the variance in each local patch is a 3x3 symmetric matrix:
    //           rr, rg, rb
    //   Sigma = rg, gg, gb
    //           rb, gb, bb

	//Calculation of var_I moved to preprocess()

	central_filter(mean_I, &mean_cv, var_I, cov_Ip, a, globalWorksize_3D);

//	if(imgType == CV_32F){
		boxfilter(&a[0], &tmp_3DA_r, globalWorksize_bf_3D);
		boxfilter(&a[1], &tmp_3DA_g, globalWorksize_bf_3D);
		boxfilter(&a[2], &tmp_3DA_b, globalWorksize_bf_3D);
		boxfilter(&mean_cv, cl_costVol, globalWorksize_bf_3D);
//	} else if(imgType == CV_8U){
//		boxfilter(&a[0], &bf3Dtmp, &tmp_3DA_r, globalWorksize_bfc_3D);
//		boxfilter(&a[1], &bf3Dtmp, &tmp_3DA_g, globalWorksize_bfc_3D);
//		boxfilter(&a[2], &bf3Dtmp, &tmp_3DA_b, globalWorksize_bfc_3D);
//		boxfilter(&mean_cv, &bf3Dtmp, cl_costVol, globalWorksize_bfc_3D);
//	}

	elementwiseMulDD(Ir, &tmp_3DA_r, &tmp_3DA_r);
	elementwiseMulDD(Ig, &tmp_3DA_g, &tmp_3DA_g);
	elementwiseMulDD(Ib, &tmp_3DA_b, &tmp_3DA_b);
	add(&tmp_3DA_r, &tmp_3DA_g, &tmp_3DA_r, globalWorksize_3D);
	add(&tmp_3DA_r, &tmp_3DA_b, &tmp_3DA_r, globalWorksize_3D);
	add(&tmp_3DA_r, cl_costVol, cl_costVol, globalWorksize_3D);

    return 0;
}

int CVF_cl::elementwiseMulDD(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out)
{
	int arg_num = 0;
    /* Setup the kernel arguments. */
    bool setKernelArgumentsSuccess = true;
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_mmdd, arg_num++, sizeof(cl_mem), cl_in_a));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_mmdd, arg_num++, sizeof(cl_mem), cl_in_b));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_mmdd, arg_num++, sizeof(cl_int), &width));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_mmdd, arg_num++, sizeof(cl_int), &height));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_mmdd, arg_num++, sizeof(cl_mem), cl_out));
    if (!setKernelArgumentsSuccess)
    {
		cleanUpOpenCL(*context, *commandQueue, program, kernel_mmdd, NULL, 0);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    if(OCL_STATS) printf("CVF_cl: Running MMDD Kernels\n");
	/* Enqueue the kernel */
	if (!checkSuccess(clEnqueueNDRangeKernel(*commandQueue, kernel_mmdd, 3, NULL, globalWorksize_3D, NULL, 0, NULL, &event)))
	{
		cleanUpOpenCL(*context, *commandQueue, program, kernel_mmdd, NULL, 0);
		cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	/* Wait for completion */
	if (!checkSuccess(clFinish(*commandQueue)))
	{
		cleanUpOpenCL(*context, *commandQueue, program, kernel_mmdd, NULL, 0);
		cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	if(OCL_STATS) printProfilingInfo(event);
    /* Release the event object. */
    if (!checkSuccess(clReleaseEvent(event)))
    {
        cleanUpOpenCL(*context, *commandQueue, program, kernel_mmdd, NULL, 0);
        cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    return 0;
}

int CVF_cl::elementwiseMulSD(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out, size_t *globalworksize)
{
	int arg_num = 0;
    /* Setup the kernel arguments. */
    bool setKernelArgumentsSuccess = true;
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_mmsd, arg_num++, sizeof(cl_mem), cl_in_a));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_mmsd, arg_num++, sizeof(cl_mem), cl_in_b));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_mmsd, arg_num++, sizeof(cl_int), &width));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_mmsd, arg_num++, sizeof(cl_int), &height));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_mmsd, arg_num++, sizeof(cl_mem), cl_out));
    if (!setKernelArgumentsSuccess)
    {
		cleanUpOpenCL(*context, *commandQueue, program, kernel_mmsd, NULL, 0);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    if(OCL_STATS) printf("CVF_cl: Running MMSD Kernels\n");
	/* Enqueue the kernel */
	if (!checkSuccess(clEnqueueNDRangeKernel(*commandQueue, kernel_mmsd, 3, NULL, globalworksize, NULL, 0, NULL, &event)))
	{
		cleanUpOpenCL(*context, *commandQueue, program, kernel_mmsd, NULL, 0);
		cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	/* Wait for completion */
	if (!checkSuccess(clFinish(*commandQueue)))
	{
		cleanUpOpenCL(*context, *commandQueue, program, kernel_mmsd, NULL, 0);
		cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	if(OCL_STATS) printProfilingInfo(event);
    /* Release the event object. */
    if (!checkSuccess(clReleaseEvent(event)))
    {
        cleanUpOpenCL(*context, *commandQueue, program, kernel_mmsd, NULL, 0);
        cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    return 0;
}

int CVF_cl::elementwiseDivSD(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out, size_t *globalworksize)
{
	int arg_num = 0;
    /* Setup the kernel arguments. */
    bool setKernelArgumentsSuccess = true;
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_mdsd, arg_num++, sizeof(cl_mem), cl_in_a));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_mdsd, arg_num++, sizeof(cl_mem), cl_in_b));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_mdsd, arg_num++, sizeof(cl_int), &width));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_mdsd, arg_num++, sizeof(cl_int), &height));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_mdsd, arg_num++, sizeof(cl_mem), cl_out));
    if (!setKernelArgumentsSuccess)
    {
		cleanUpOpenCL(*context, *commandQueue, program, kernel_mdsd, NULL, 0);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    if(OCL_STATS) printf("CVF_cl: Running MDSD Kernels\n");
	/* Enqueue the kernel */
	if (!checkSuccess(clEnqueueNDRangeKernel(*commandQueue, kernel_mdsd, 3, NULL, globalworksize, NULL, 0, NULL, &event)))
	{
		cleanUpOpenCL(*context, *commandQueue, program, kernel_mdsd, NULL, 0);
		cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	/* Wait for completion */
	if (!checkSuccess(clFinish(*commandQueue)))
	{
		cleanUpOpenCL(*context, *commandQueue, program, kernel_mdsd, NULL, 0);
		cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	if(OCL_STATS) printProfilingInfo(event);
    /* Release the event object. */
    if (!checkSuccess(clReleaseEvent(event)))
    {
        cleanUpOpenCL(*context, *commandQueue, program, kernel_mdsd, NULL, 0);
        cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    return 0;
}

int CVF_cl::split(cl_mem *cl_I, cl_mem *cl_Ir, cl_mem *cl_Ig, cl_mem *cl_Ib)
{
	int arg_num = 0;
	/* Setup the kernel arguments. */
	bool setKernelArgumentsSuccess = true;
	setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_split, arg_num++, sizeof(cl_mem), cl_I));
	setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_split, arg_num++, sizeof(cl_int), &width));
	setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_split, arg_num++, sizeof(cl_mem), cl_Ir));
	setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_split, arg_num++, sizeof(cl_mem), cl_Ig));
	setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_split, arg_num++, sizeof(cl_mem), cl_Ib));

	if (!setKernelArgumentsSuccess)
	{
	   cleanUpOpenCL(*context, *commandQueue, program, kernel_split, NULL, 0);
		cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
	}


    if(OCL_STATS) printf("CVF_cl: Running Split Kernels\n");
	/* Enqueue the kernel */
	if (!checkSuccess(clEnqueueNDRangeKernel(*commandQueue, kernel_split, 2, NULL, globalWorksize_split, NULL, 0, NULL, &event)))
	{
	   cleanUpOpenCL(*context, *commandQueue, program, kernel_split, NULL, 0);
		cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	/* Wait for completion */
	if (!checkSuccess(clFinish(*commandQueue)))
	{
	   cleanUpOpenCL(*context, *commandQueue, program, kernel_split, NULL, 0);
		cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	if(OCL_STATS) printProfilingInfo(event);
    /* Release the event object. */
    if (!checkSuccess(clReleaseEvent(event)))
    {
        cleanUpOpenCL(*context, *commandQueue, program, kernel_split, NULL, 0);
        cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    return 0;
}

int CVF_cl::sub(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out, size_t *globalworksize)
{
	int arg_num = 0;
    /* Setup the kernel arguments. */
    bool setKernelArgumentsSuccess = true;
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_sub, arg_num++, sizeof(cl_mem), cl_in_a));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_sub, arg_num++, sizeof(cl_mem), cl_in_b));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_sub, arg_num++, sizeof(cl_int), &width));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_sub, arg_num++, sizeof(cl_int), &height));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_sub, arg_num++, sizeof(cl_mem), cl_out));
    if (!setKernelArgumentsSuccess)
    {
		cleanUpOpenCL(*context, *commandQueue, program, kernel_sub, NULL, 0);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    if(OCL_STATS) printf("CVF_cl: Running Sub Kernels\n");
	/* Enqueue the kernel */
	if (!checkSuccess(clEnqueueNDRangeKernel(*commandQueue, kernel_sub, 3, NULL, globalworksize, NULL, 0, NULL, &event)))
	{
		cleanUpOpenCL(*context, *commandQueue, program, kernel_sub, NULL, 0);
		cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	/* Wait for completion */
	if (!checkSuccess(clFinish(*commandQueue)))
	{
		cleanUpOpenCL(*context, *commandQueue, program, kernel_sub, NULL, 0);
		cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	if(OCL_STATS) printProfilingInfo(event);
    /* Release the event object. */
    if (!checkSuccess(clReleaseEvent(event)))
    {
        cleanUpOpenCL(*context, *commandQueue, program, kernel_sub, NULL, 0);
        cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    return 0;
}

int CVF_cl::add(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out, size_t *globalworksize)
{
	int arg_num = 0;
    /* Setup the kernel arguments. */
    bool setKernelArgumentsSuccess = true;
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_add, arg_num++, sizeof(cl_mem), cl_in_a));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_add, arg_num++, sizeof(cl_mem), cl_in_b));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_add, arg_num++, sizeof(cl_int), &width));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_add, arg_num++, sizeof(cl_int), &height));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_add, arg_num++, sizeof(cl_mem), cl_out));
    if (!setKernelArgumentsSuccess)
    {
		cleanUpOpenCL(*context, *commandQueue, program, kernel_add, NULL, 0);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    if(OCL_STATS) printf("CVF_cl: Running Add Kernels\n");
	/* Enqueue the kernel */
	if (!checkSuccess(clEnqueueNDRangeKernel(*commandQueue, kernel_add, 3, NULL, globalworksize, NULL, 0, NULL, &event)))
	{
		cleanUpOpenCL(*context, *commandQueue, program, kernel_add, NULL, 0);
		cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	/* Wait for completion */
	if (!checkSuccess(clFinish(*commandQueue)))
	{
		cleanUpOpenCL(*context, *commandQueue, program, kernel_add, NULL, 0);
		cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	if(OCL_STATS) printProfilingInfo(event);
    /* Release the event object. */
    if (!checkSuccess(clReleaseEvent(event)))
    {
        cleanUpOpenCL(*context, *commandQueue, program, kernel_add, NULL, 0);
        cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    return 0;
}

int CVF_cl::central_filter(cl_mem *mean_I_in, cl_mem *mean_cv_io, cl_mem *var_I_in, cl_mem *cov_Ip_in,  cl_mem *a_out, size_t *globalworksize)
{
	int arg_num = 0;
    /* Setup the kernel arguments. */
    bool setKernelArgumentsSuccess = true;
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_centf, arg_num++, sizeof(cl_mem), &mean_I_in[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_centf, arg_num++, sizeof(cl_mem), &mean_I_in[1]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_centf, arg_num++, sizeof(cl_mem), &mean_I_in[2]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_centf, arg_num++, sizeof(cl_mem), &var_I_in[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_centf, arg_num++, sizeof(cl_mem), &var_I_in[1]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_centf, arg_num++, sizeof(cl_mem), &var_I_in[2]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_centf, arg_num++, sizeof(cl_mem), &var_I_in[3]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_centf, arg_num++, sizeof(cl_mem), &var_I_in[4]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_centf, arg_num++, sizeof(cl_mem), &var_I_in[5]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_centf, arg_num++, sizeof(cl_mem), &cov_Ip_in[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_centf, arg_num++, sizeof(cl_mem), &cov_Ip_in[1]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_centf, arg_num++, sizeof(cl_mem), &cov_Ip_in[2]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_centf, arg_num++, sizeof(cl_int), &width));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_centf, arg_num++, sizeof(cl_int), &height));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_centf, arg_num++, sizeof(cl_mem), &a_out[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_centf, arg_num++, sizeof(cl_mem), &a_out[1]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_centf, arg_num++, sizeof(cl_mem), &a_out[2]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_centf, arg_num++, sizeof(cl_mem), mean_cv_io));
    if (!setKernelArgumentsSuccess)
    {
		cleanUpOpenCL(*context, *commandQueue, program, kernel_centf, NULL, 0);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    if(OCL_STATS) printf("CVF_cl: Running central filter Kernels\n");
	/* Enqueue the kernel */
	if (!checkSuccess(clEnqueueNDRangeKernel(*commandQueue, kernel_centf, 3, NULL, globalworksize, NULL, 0, NULL, &event)))
	{
		cleanUpOpenCL(*context, *commandQueue, program, kernel_centf, NULL, 0);
		cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	/* Wait for completion */
	if (!checkSuccess(clFinish(*commandQueue)))
	{
		cleanUpOpenCL(*context, *commandQueue, program, kernel_centf, NULL, 0);
		cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	if(OCL_STATS) printProfilingInfo(event);
    /* Release the event object. */
    if (!checkSuccess(clReleaseEvent(event)))
    {
        cleanUpOpenCL(*context, *commandQueue, program, kernel_centf, NULL, 0);
        cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    return 0;
}

int CVF_cl::preproc_maths(cl_mem *mean_I_in, cl_mem *mean_Ixx_in, cl_mem *var_I_out, size_t *globalworksize)
{
	int arg_num = 0;
    /* Setup the kernel arguments. */
    bool setKernelArgumentsSuccess = true;
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_var, arg_num++, sizeof(cl_mem), &mean_I_in[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_var, arg_num++, sizeof(cl_mem), &mean_I_in[1]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_var, arg_num++, sizeof(cl_mem), &mean_I_in[2]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_var, arg_num++, sizeof(cl_mem), &mean_Ixx_in[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_var, arg_num++, sizeof(cl_mem), &mean_Ixx_in[1]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_var, arg_num++, sizeof(cl_mem), &mean_Ixx_in[2]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_var, arg_num++, sizeof(cl_mem), &mean_Ixx_in[3]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_var, arg_num++, sizeof(cl_mem), &mean_Ixx_in[4]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_var, arg_num++, sizeof(cl_mem), &mean_Ixx_in[5]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_var, arg_num++, sizeof(cl_int), &width));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_var, arg_num++, sizeof(cl_int), &height));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_var, arg_num++, sizeof(cl_mem), &var_I_out[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_var, arg_num++, sizeof(cl_mem), &var_I_out[1]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_var, arg_num++, sizeof(cl_mem), &var_I_out[2]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_var, arg_num++, sizeof(cl_mem), &var_I_out[3]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_var, arg_num++, sizeof(cl_mem), &var_I_out[4]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_var, arg_num++, sizeof(cl_mem), &var_I_out[5]));
    if (!setKernelArgumentsSuccess)
    {
		cleanUpOpenCL(*context, *commandQueue, program, kernel_var, NULL, 0);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    if(OCL_STATS) printf("CVF_cl: Running variance maths Kernels\n");
	/* Enqueue the kernel */
	if (!checkSuccess(clEnqueueNDRangeKernel(*commandQueue, kernel_var, 3, NULL, globalworksize, NULL, 0, NULL, &event)))
	{
		cleanUpOpenCL(*context, *commandQueue, program, kernel_var, NULL, 0);
		cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	/* Wait for completion */
	if (!checkSuccess(clFinish(*commandQueue)))
	{
		cleanUpOpenCL(*context, *commandQueue, program, kernel_var, NULL, 0);
		cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	if(OCL_STATS) printProfilingInfo(event);
    /* Release the event object. */
    if (!checkSuccess(clReleaseEvent(event)))
    {
        cleanUpOpenCL(*context, *commandQueue, program, kernel_var, NULL, 0);
        cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    return 0;
}

int CVF_cl::boxfilter(cl_mem *cl_in, cl_mem *cl_out, size_t *globalworksize)
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

    if(OCL_STATS) printf("CVF_cl: Running boxfilter Kernels\n");
	/* Enqueue the kernel */
	if (!checkSuccess(clEnqueueNDRangeKernel(*commandQueue, kernel_bf, 3, NULL, globalworksize, NULL, 0, NULL, &event)))
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

	if(OCL_STATS) printProfilingInfo(event);
    /* Release the event object. */
    if (!checkSuccess(clReleaseEvent(event)))
    {
        cleanUpOpenCL(*context, *commandQueue, program, kernel_bf, NULL, 0);
        cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    return 0;
}

int CVF_cl::boxfilter(cl_mem *cl_in, cl_mem *cl_tmp, cl_mem *cl_out, size_t *globalworksize)
{
	int iRadius = 4;
	float fScale = 1.0f/(2.0f * iRadius + 1.0f);

	int arg_num = 0;
    /* Setup the kernel arguments. */
    bool setKernelArgumentsSuccess = true;
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_bfc_rows, arg_num++, sizeof(cl_mem), cl_in));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_bfc_rows, arg_num++, sizeof(cl_mem), cl_tmp));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_bfc_rows, arg_num++, sizeof(cl_int), &width));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_bfc_rows, arg_num++, sizeof(cl_int), &height));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_bfc_rows, arg_num++, sizeof(cl_int), &iRadius));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_bfc_rows, arg_num++, sizeof(cl_float), &fScale));
    if (!setKernelArgumentsSuccess)
    {
		cleanUpOpenCL(*context, *commandQueue, program, kernel_bfc_rows, NULL, 0);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
    }

	arg_num = 0;
    /* Setup the kernel arguments. */
    setKernelArgumentsSuccess = true;
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_bfc_cols, arg_num++, sizeof(cl_mem), cl_tmp));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_bfc_cols, arg_num++, sizeof(cl_mem), cl_out));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_bfc_cols, arg_num++, sizeof(cl_int), &width));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_bfc_cols, arg_num++, sizeof(cl_int), &height));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_bfc_cols, arg_num++, sizeof(cl_int), &iRadius));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_bfc_cols, arg_num++, sizeof(cl_float), &fScale));
    if (!setKernelArgumentsSuccess)
    {
		cleanUpOpenCL(*context, *commandQueue, program, kernel_bfc_rows, NULL, 0);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    if(OCL_STATS) printf("CVF_cl: Running boxfilter row Kernels\n");
	/* Enqueue the kernel */
	if (!checkSuccess(clEnqueueNDRangeKernel(*commandQueue, kernel_bfc_rows, 3, NULL, globalworksize, NULL, 0, NULL, &event)))
	{
		cleanUpOpenCL(*context, *commandQueue, program, kernel_bfc_rows, NULL, 0);
		cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	globalworksize[0] = (size_t)width;

    if(OCL_STATS) printf("CVF_cl: Running boxfilter col Kernels\n");
	/* Enqueue the kernel */
	if (!checkSuccess(clEnqueueNDRangeKernel(*commandQueue, kernel_bfc_cols, 3, NULL, globalworksize, NULL, 0, NULL, &event)))
	{
		cleanUpOpenCL(*context, *commandQueue, program, kernel_bfc_cols, NULL, 0);
		cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}
    return 0;
}
