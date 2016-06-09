/*---------------------------------------------------------------------------
   CVF_cl.cpp - OpenCL Cost Volume Filter Code
              - Guided Image Filter
              - Box Filter
  ---------------------------------------------------------------------------
   Editor: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "CVF_cl.h"

float get_rt_cvf_cl(){
	struct timespec realtime;
	clock_gettime(CLOCK_MONOTONIC,&realtime);
	return (float)(realtime.tv_sec*1000000+realtime.tv_nsec/1000);
}

CVF_cl::CVF_cl(cl_context* context, cl_command_queue* commandQueue, cl_device_id device, Mat I, const int d) :
				context(context), commandQueue(commandQueue), dispRange(d)
{
	//OpenCL Setup
    program = 0;

    if (!createProgram(*context, device, "assets/cvf.cl", &program))
    {
        cleanUpOpenCL(*context, *commandQueue, program, kernel_split, NULL, 0);
        cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
    }

	kernel_mmsd = clCreateKernel(program, "EWMul_SameDim", &errorNumber);
	kernel_mmdd = clCreateKernel(program, "EWMul_DiffDim", &errorNumber);
	kernel_mdsd = clCreateKernel(program, "EWDiv_SameDim", &errorNumber);
	kernel_split = clCreateKernel(program, "Split", &errorNumber);
	kernel_sub = clCreateKernel(program, "Subtract", &errorNumber);
	kernel_add = clCreateKernel(program, "Add", &errorNumber);
	kernel_add_const = clCreateKernel(program, "Add_Const", &errorNumber);
	kernel_bf = clCreateKernel(program, "boxfilter", &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        cleanUpOpenCL(*context, *commandQueue, program, NULL, NULL, 0);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    /* An event to associate with the Kernel. Allows us to retreive profiling information later. */
    event = 0;

	width = I.cols;
	height = I.rows;
    channels = I.channels();

	//Buffers in accending size order
	bufferSize_2D_C1F = width * height * sizeof(cl_float);
	bufferSize_2D_C3F = width * height * sizeof(cl_float) * channels;
	bufferSize_3D_C1F = width * height * dispRange * sizeof(cl_float);

    globalWorksize_3D[0] = (size_t)width;
    globalWorksize_3D[1] = (size_t)height;
    globalWorksize_3D[2] = (size_t)dispRange;

    globalWorksize_2D[0] = (size_t)width;
    globalWorksize_2D[1] = (size_t)height;
    globalWorksize_2D[2] = (size_t)1;

    globalWorksize_split[0] = (size_t)width/3;
    globalWorksize_split[1] = (size_t)height;

    globalWorksize_bf_3D[0] = (size_t)width/16;
    globalWorksize_bf_3D[1] = (size_t)height;
    globalWorksize_bf_3D[2] = (size_t)dispRange;

    globalWorksize_bf_2D[0] = (size_t)width/16;
    globalWorksize_bf_2D[1] = (size_t)height;
    globalWorksize_bf_2D[2] = (size_t)1;

	bool createMemoryObjectsSuccess = true;
	Ir = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	Ig = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	Ib = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);

	mean_Ir = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	mean_Ig = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	mean_Ib = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);

	Irr = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	Irg = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	Irb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	Igg = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	Igb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	Ibb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);

	mean_Irr = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	mean_Irg = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	mean_Irb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	mean_Igg = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	mean_Igb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	mean_Ibb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);

	var_Irr = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	var_Irg = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	var_Irb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	var_Igg = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	var_Igb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	var_Ibb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);

	inv_rr = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	inv_rg = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	inv_rb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	inv_gg = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	inv_gb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	inv_bb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);

	inv_ggbb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	inv_gbrb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	inv_rggb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	inv_rrbb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	inv_rbrg = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	inv_rrgg = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);

	inv_gbgb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	inv_rgbb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	inv_ggrb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	inv_rbrb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	inv_rrgb = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	inv_rgrg = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);

	covDet = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_2D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);

	mean_cv = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);

	b = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	mean_b = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);

	a_r = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	a_g = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	a_b = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);

	tmp_3DA_r = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	tmp_3DA_g = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	tmp_3DA_b = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);

	tmp_3DB_r = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	tmp_3DB_g = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);
	tmp_3DB_b = clCreateBuffer(*context, CL_MEM_READ_WRITE, bufferSize_3D_C1F, NULL, &errorNumber);
	createMemoryObjectsSuccess &= checkSuccess(errorNumber);

	if (!createMemoryObjectsSuccess)
	{
		cerr << "Failed to create OpenCL buffers. " << __FILE__ << ":"<< __LINE__ << endl;
	}
    printf("Allocated OpenCL Buffers\n");
}

CVF_cl::~CVF_cl(void)
{
	clReleaseMemObject(Ir);
	clReleaseMemObject(Ig);
	clReleaseMemObject(Ib);
	clReleaseMemObject(mean_Ir);
	clReleaseMemObject(mean_Ig);
	clReleaseMemObject(mean_Ib);
	clReleaseMemObject(Irr);
	clReleaseMemObject(Irg);
	clReleaseMemObject(Irb);
	clReleaseMemObject(Igg);
	clReleaseMemObject(Igb);
	clReleaseMemObject(Ibb);
	clReleaseMemObject(mean_Irr);
	clReleaseMemObject(mean_Irg);
	clReleaseMemObject(mean_Irb);
	clReleaseMemObject(mean_Igg);
	clReleaseMemObject(mean_Igb);
	clReleaseMemObject(mean_Ibb);
	clReleaseMemObject(var_Irr);
	clReleaseMemObject(var_Irg);
	clReleaseMemObject(var_Irb);
	clReleaseMemObject(var_Igg);
	clReleaseMemObject(var_Igb);
	clReleaseMemObject(var_Ibb);
	clReleaseMemObject(inv_rr);
	clReleaseMemObject(inv_rg);
	clReleaseMemObject(inv_rb);
	clReleaseMemObject(inv_gg);
	clReleaseMemObject(inv_gb);
	clReleaseMemObject(inv_bb);

	clReleaseMemObject(covDet);

	clReleaseMemObject(b);
	clReleaseMemObject(mean_b);
	clReleaseMemObject(mean_cv);
	clReleaseMemObject(tmp_3DA_r);
	clReleaseMemObject(tmp_3DA_g);
	clReleaseMemObject(tmp_3DA_b);
	clReleaseMemObject(tmp_3DB_r);
	clReleaseMemObject(tmp_3DB_g);
	clReleaseMemObject(tmp_3DB_b);
}


int CVF_cl::filterCV(cl_mem* cl_Img, cl_mem* cl_costVol)
{
//    float time_start = get_rt_cvf_cl();
//    printf("0 Time from start = %.0f us\n", get_rt_cvf_cl() - time_start);

    split(cl_Img, &Ir, &Ig, &Ib);

//	printf("1 Time from start = %.0f us\n", get_rt_cvf_cl() - time_start);

	boxfilter(&Ir, &mean_Ir, globalWorksize_bf_2D);
	boxfilter(&Ig, &mean_Ig, globalWorksize_bf_2D);
	boxfilter(&Ib, &mean_Ib, globalWorksize_bf_2D);

//	printf("2 Time from start = %.0f us\n", get_rt_cvf_cl() - time_start);

	boxfilter(cl_costVol, &mean_cv, globalWorksize_bf_3D);

//    printf("3 Time from start = %.0f us\n", get_rt_cvf_cl() - time_start);

	elementwiseMulDD(&Ir, cl_costVol, &tmp_3DA_r);
	elementwiseMulDD(&Ig, cl_costVol, &tmp_3DA_g);
	elementwiseMulDD(&Ib, cl_costVol, &tmp_3DA_b);

//	printf("4 Time from start = %.0f us\n", get_rt_cvf_cl() - time_start);

	boxfilter(&tmp_3DA_r, &tmp_3DB_r, globalWorksize_bf_3D);
	boxfilter(&tmp_3DA_g, &tmp_3DB_g, globalWorksize_bf_3D);
	boxfilter(&tmp_3DA_b, &tmp_3DB_b, globalWorksize_bf_3D);

//	printf("5 Time from start = %.0f us\n", get_rt_cvf_cl() - time_start);
//	exit(1);

	elementwiseMulDD(&mean_Ir, &mean_cv, &tmp_3DA_r);
	elementwiseMulDD(&mean_Ig, &mean_cv, &tmp_3DA_g);
	elementwiseMulDD(&mean_Ib, &mean_cv, &tmp_3DA_b);

	sub(&tmp_3DB_r, &tmp_3DA_r, &tmp_3DA_r, globalWorksize_3D);
	sub(&tmp_3DB_g, &tmp_3DA_g, &tmp_3DA_g, globalWorksize_3D);
	sub(&tmp_3DB_b, &tmp_3DA_b, &tmp_3DA_b, globalWorksize_3D);

    // variance of I in each local patch: the matrix Sigma in Eqn (14).
    // Note the variance in each local patch is a 3x3 symmetric matrix:
    //           rr, rg, rb
    //   Sigma = rg, gg, gb
    //           rb, gb, bb
	elementwiseMulSD(&Ir, &Ir, &Irr, globalWorksize_2D);
	elementwiseMulSD(&Ir, &Ig, &Irg, globalWorksize_2D);
	elementwiseMulSD(&Ir, &Ib, &Irb, globalWorksize_2D);
	elementwiseMulSD(&Ig, &Ig, &Igg, globalWorksize_2D);
	elementwiseMulSD(&Ig, &Ib, &Igb, globalWorksize_2D);
	elementwiseMulSD(&Ib, &Ib, &Ibb, globalWorksize_2D);

	boxfilter(&Irr, &mean_Irr, globalWorksize_bf_2D);
	boxfilter(&Irg, &mean_Irg, globalWorksize_bf_2D);
	boxfilter(&Irb, &mean_Irb, globalWorksize_bf_2D);
	boxfilter(&Igg, &mean_Igg, globalWorksize_bf_2D);
	boxfilter(&Igb, &mean_Igb, globalWorksize_bf_2D);
	boxfilter(&Ibb, &mean_Ibb, globalWorksize_bf_2D);

	elementwiseMulSD(&mean_Ir, &mean_Ir, &var_Irr, globalWorksize_2D);
	elementwiseMulSD(&mean_Ir, &mean_Ig, &var_Irg, globalWorksize_2D);
	elementwiseMulSD(&mean_Ir, &mean_Ib, &var_Irb, globalWorksize_2D);
	elementwiseMulSD(&mean_Ig, &mean_Ig, &var_Igg, globalWorksize_2D);
	elementwiseMulSD(&mean_Ig, &mean_Ib, &var_Igb, globalWorksize_2D);
	elementwiseMulSD(&mean_Ib, &mean_Ib, &var_Ibb, globalWorksize_2D);

	sub(&mean_Irr, &var_Irr, &var_Irr, globalWorksize_2D);
	sub(&mean_Irg, &var_Irg, &var_Irg, globalWorksize_2D);
	sub(&mean_Irb, &var_Irb, &var_Irb, globalWorksize_2D);
	sub(&mean_Igg, &var_Igg, &var_Igg, globalWorksize_2D);
	sub(&mean_Igb, &var_Igb, &var_Igb, globalWorksize_2D);
	sub(&mean_Ibb, &var_Ibb, &var_Ibb, globalWorksize_2D);

	add_constant(&var_Irr, EPS, &var_Irr);
	add_constant(&var_Igg, EPS, &var_Igg);
	add_constant(&var_Ibb, EPS, &var_Ibb);

    // Inverse of Sigma + eps * I
	elementwiseMulSD(&var_Igg, &var_Ibb, &inv_ggbb, globalWorksize_2D);	elementwiseMulSD(&var_Igb, &var_Igb, &inv_gbgb, globalWorksize_2D);
	elementwiseMulSD(&var_Igb, &var_Irb, &inv_gbrb, globalWorksize_2D);	elementwiseMulSD(&var_Irg, &var_Ibb, &inv_rgbb, globalWorksize_2D);
	elementwiseMulSD(&var_Irg, &var_Igb, &inv_rggb, globalWorksize_2D);	elementwiseMulSD(&var_Igg, &var_Irb, &inv_ggrb, globalWorksize_2D);
	elementwiseMulSD(&var_Irr, &var_Ibb, &inv_rrbb, globalWorksize_2D);	elementwiseMulSD(&var_Irb, &var_Irb, &inv_rbrb, globalWorksize_2D);
	elementwiseMulSD(&var_Irb, &var_Irg, &inv_rbrg, globalWorksize_2D);	elementwiseMulSD(&var_Irr, &var_Igb, &inv_rrgb, globalWorksize_2D);
	elementwiseMulSD(&var_Irr, &var_Igg, &inv_rrgg, globalWorksize_2D);	elementwiseMulSD(&var_Irg, &var_Irg, &inv_rgrg, globalWorksize_2D);

	sub(&inv_ggbb, &inv_gbgb, &inv_rr, globalWorksize_2D);
	sub(&inv_gbrb, &inv_rgbb, &inv_rg, globalWorksize_2D);
	sub(&inv_rggb, &inv_ggrb, &inv_rb, globalWorksize_2D);
	sub(&inv_rrbb, &inv_rbrb, &inv_gg, globalWorksize_2D);
	sub(&inv_rbrg, &inv_rrgb, &inv_gb, globalWorksize_2D);
	sub(&inv_rrgg, &inv_rgrg, &inv_bb, globalWorksize_2D);

	elementwiseMulSD(&var_Irr, &inv_rr, &var_Irr, globalWorksize_2D);
	elementwiseMulSD(&var_Irg, &inv_rg, &var_Irg, globalWorksize_2D);
	elementwiseMulSD(&var_Irb, &inv_rb, &var_Irb, globalWorksize_2D);
	add(&var_Irr, &var_Irg, &var_Irr, globalWorksize_2D);
	add(&var_Irb, &var_Irr, &covDet, globalWorksize_2D);

	elementwiseDivSD(&inv_rr, &covDet, &inv_rr, globalWorksize_2D);
	elementwiseDivSD(&inv_rg, &covDet, &inv_rg, globalWorksize_2D);
	elementwiseDivSD(&inv_rb, &covDet, &inv_rb, globalWorksize_2D);
	elementwiseDivSD(&inv_gg, &covDet, &inv_gg, globalWorksize_2D);
	elementwiseDivSD(&inv_gb, &covDet, &inv_gb, globalWorksize_2D);
	elementwiseDivSD(&inv_bb, &covDet, &inv_bb, globalWorksize_2D);

	elementwiseMulDD(&inv_rr, &tmp_3DA_r, &tmp_3DB_r);
	elementwiseMulDD(&inv_rg, &tmp_3DA_g, &tmp_3DB_g);
	elementwiseMulDD(&inv_rb, &tmp_3DA_b, &tmp_3DB_b);
	add(&tmp_3DB_r, &tmp_3DB_g, &tmp_3DB_r, globalWorksize_3D);
	add(&tmp_3DB_r, &tmp_3DB_b, &a_r, globalWorksize_3D);

	elementwiseMulDD(&inv_rg, &tmp_3DA_r, &tmp_3DB_r);
	elementwiseMulDD(&inv_gg, &tmp_3DA_g, &tmp_3DB_g);
	elementwiseMulDD(&inv_gb, &tmp_3DA_b, &tmp_3DB_b);
	add(&tmp_3DB_r, &tmp_3DB_g, &tmp_3DB_r, globalWorksize_3D);
	add(&tmp_3DB_r, &tmp_3DB_b, &a_g, globalWorksize_3D);

	elementwiseMulDD(&inv_rb, &tmp_3DA_r, &tmp_3DB_r);
	elementwiseMulDD(&inv_gb, &tmp_3DA_g, &tmp_3DB_g);
	elementwiseMulDD(&inv_bb, &tmp_3DA_b, &tmp_3DB_b);
	add(&tmp_3DB_r, &tmp_3DB_g, &tmp_3DB_r, globalWorksize_3D);
	add(&tmp_3DB_r, &tmp_3DB_b, &a_b, globalWorksize_3D);

	elementwiseMulDD(&mean_Ir, &a_r, &tmp_3DB_r);
	elementwiseMulDD(&mean_Ig, &a_g, &tmp_3DB_g);
	elementwiseMulDD(&mean_Ib, &a_b, &tmp_3DB_b);

	sub(&mean_cv, &tmp_3DB_r, &mean_cv, globalWorksize_3D);
	sub(&mean_cv, &tmp_3DB_g, &mean_cv, globalWorksize_3D);
	sub(&mean_cv, &tmp_3DB_b, &b, globalWorksize_3D);

	boxfilter(&a_r, &tmp_3DA_r, globalWorksize_bf_3D);
	boxfilter(&a_g, &tmp_3DA_g, globalWorksize_bf_3D);
	boxfilter(&a_b, &tmp_3DA_b, globalWorksize_bf_3D);
	boxfilter(&b, &mean_b, globalWorksize_bf_3D);

	elementwiseMulDD(&Ir, &tmp_3DA_r, &tmp_3DA_r);
	elementwiseMulDD(&Ig, &tmp_3DA_g, &tmp_3DA_g);
	elementwiseMulDD(&Ib, &tmp_3DA_b, &tmp_3DA_b);
	add(&tmp_3DA_r, &tmp_3DA_g, &tmp_3DA_r, globalWorksize_3D);
	add(&tmp_3DA_r, &tmp_3DA_b, &tmp_3DA_r, globalWorksize_3D);
	add(&tmp_3DA_r, &mean_b, cl_costVol, globalWorksize_3D);

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

int CVF_cl::add_constant(cl_mem *cl_in_a, float const_add, cl_mem *cl_out)
{
	int arg_num = 0;
    /* Setup the kernel arguments. */
    bool setKernelArgumentsSuccess = true;
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_add_const, arg_num++, sizeof(cl_mem), cl_in_a));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_add_const, arg_num++, sizeof(cl_float), &const_add));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_add_const, arg_num++, sizeof(cl_int), &width));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_add_const, arg_num++, sizeof(cl_int), &height));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel_add_const, arg_num++, sizeof(cl_mem), cl_out));
    if (!setKernelArgumentsSuccess)
    {
		cleanUpOpenCL(*context, *commandQueue, program, kernel_add_const, NULL, 0);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    if(OCL_STATS) printf("CVF_cl: Running Add_const Kernels\n");
	/* Enqueue the kernel */
	if (!checkSuccess(clEnqueueNDRangeKernel(*commandQueue, kernel_add_const, 3, NULL, globalWorksize_2D, NULL, 0, NULL, &event)))
	{
		cleanUpOpenCL(*context, *commandQueue, program, kernel_add_const, NULL, 0);
		cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	/* Wait for completion */
	if (!checkSuccess(clFinish(*commandQueue)))
	{
		cleanUpOpenCL(*context, *commandQueue, program, kernel_add_const, NULL, 0);
		cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
		return 1;
	}

	if(OCL_STATS) printProfilingInfo(event);
    /* Release the event object. */
    if (!checkSuccess(clReleaseEvent(event)))
    {
        cleanUpOpenCL(*context, *commandQueue, program, kernel_add_const, NULL, 0);
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
