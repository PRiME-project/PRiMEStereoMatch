/*---------------------------------------------------------------------------
   CVF_cl.h - OpenCL Cost Volume Filter Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "common.h"

#define R_WIN 9
#define EPS 0.0001f

//
// GIF for Cost Computation
//
class CVF_cl
{
public:
	//OpenCL Variables
    cl_context* context;
	cl_command_queue* commandQueue;
    cl_program program;
    cl_kernel kernel_mmsd, kernel_mmdd, kernel_mdsd, kernel_bf;
    cl_kernel kernel_split, kernel_sub, kernel_add, kernel_add_const;
    cl_kernel kernel_centf, kernel_var;
    cl_int errorNumber;
    cl_event event;

    cl_int width, height, channels, dispRange;
    size_t bufferSize_2D_C1F, bufferSize_2D_C3F, bufferSize_3D_C1F;

    size_t globalWorksize_3D[3], globalWorksize_2D[3];
    size_t globalWorksize_bf_3D[3], globalWorksize_bf_2D[3];
    size_t globalWorksize_split[2];

	enum buff_id {CVC_LIMG = 0, CVC_RIMG, CVC_LGRDX, CVC_RGRDX, CV_LCV, CV_RCV, DS_LDM, DS_RDM};

	cl_mem Ir, Ig, Ib;
	cl_mem mean_cv;
	cl_mem *Ixx, *mean_I, *mean_Ixx;
	cl_mem *var_I, *cov_Ip, *a;

	cl_mem tmp_3DA_r, tmp_3DA_g, tmp_3DA_b;
	cl_mem tmp_3DB_r, tmp_3DB_g, tmp_3DB_b;

	CVF_cl(cl_context* context, cl_command_queue* commandQueue, cl_device_id device, Mat I, const int d);
	~CVF_cl(void);

	int preprocess(cl_mem* cl_Img);
	int filterCV(cl_mem* cl_costVol);

	int elementwiseMulSD(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out, size_t *globalworksize);
	int elementwiseMulDD(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out);
	int elementwiseDivSD(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out, size_t *globalworksize);
    int split(cl_mem* cl_Img, cl_mem* cl_Ir, cl_mem* cl_Ig, cl_mem* cl_Ib);
    int sub(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out, size_t *globalworksize);
    int add(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out, size_t *globalworksize);
    int add_constant(cl_mem *cl_in_a, float const_add, cl_mem *cl_out);
    int boxfilter(cl_mem *cl_in, cl_mem *cl_out, size_t *globalworksize);
	int central_filter(cl_mem *mean_I_in, cl_mem *mean_cv_io, cl_mem *var_I_in, cl_mem *cov_Ip_in,  cl_mem *a_in, size_t *globalworksize);
	int preproc_maths(cl_mem *mean_I_in, cl_mem *mean_Ixx_in, cl_mem *var_I_out, size_t *globalworksize);
};
