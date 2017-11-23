/*---------------------------------------------------------------------------
   CVF_cl.h - OpenCL Cost Volume Filter Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "oclUtil.h"

#define FILE_CVF_PROG BASE_DIR "assets/cvf.cl"
#define R_WIN 9

//
// GIF for Cost Computation
//
class CVF_cl
{
public:
	CVF_cl(cl_context* context, cl_command_queue* commandQueue, cl_device_id device, Mat* I, const int d);
	~CVF_cl(void);

	int preprocess(cl_mem* Ir, cl_mem* Ig, cl_mem* Ib);
	int filterCV(cl_mem* cl_costVol);

private:
	//OpenCL Variables
    cl_context* context;
	cl_command_queue* commandQueue;
    cl_program program;
    cl_kernel kernel_mmsd, kernel_mmdd, kernel_mdsd, kernel_bf;
    cl_kernel kernel_split, kernel_sub, kernel_add;
    cl_kernel kernel_centf, kernel_var;
    cl_kernel kernel_bfc_rows, kernel_bfc_cols;
    cl_int errorNumber;
    cl_event event;

//    int imgType;
    cl_int width, height, channels, maxDis;
    size_t bufferSize_2D, bufferSize_3D;

    size_t globalWorksize_3D[3], globalWorksize_2D[3];
    size_t globalWorksize_bf_3D[3], globalWorksize_bf_2D[3];
    size_t globalWorksize_bfc_3D[3], globalWorksize_bfc_2D[3];
    size_t globalWorksize_split[2];

	cl_mem *Ir, *Ig, *Ib;
	cl_mem mean_cv;
	cl_mem *Ixx, *mean_I, *mean_Ixx;
	cl_mem *var_I, *cov_Ip, *a;

	cl_mem tmp_3DA_r, tmp_3DA_g, tmp_3DA_b;
	cl_mem tmp_3DB_r, tmp_3DB_g, tmp_3DB_b;
	//cl_mem bf2Dtmp, bf3Dtmp;

	int elementwiseMulSD(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out, size_t *globalworksize);
	int elementwiseMulDD(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out);
	int elementwiseDivSD(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out, size_t *globalworksize);
    int split(cl_mem* cl_Img, cl_mem* cl_Ir, cl_mem* cl_Ig, cl_mem* cl_Ib);
    int sub(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out, size_t *globalworksize);
    int add(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out, size_t *globalworksize);
	int central_filter(cl_mem *mean_I_in, cl_mem *mean_cv_io, cl_mem *var_I_in, cl_mem *cov_Ip_in,  cl_mem *a_in, size_t *globalworksize);
	int preproc_maths(cl_mem *mean_I_in, cl_mem *mean_Ixx_in, cl_mem *var_I_out, size_t *globalworksize);
    int boxfilter(cl_mem *cl_in, cl_mem *cl_out, size_t *globalworksize);
	int boxfilter(cl_mem *cl_in, cl_mem *cl_tmp, cl_mem *cl_out, size_t *globalworksize);
};
