/*---------------------------------------------------------------------------
   CVF_cl.h - OpenCL Cost Volume Filter Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
//#include "BoxFilter.h"
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
    cl_int errorNumber;
    cl_event event;

    cl_int width, height, channels, dispRange;
    size_t bufferSize_2D_C1F, bufferSize_2D_C3F, bufferSize_3D_C1F;

    size_t globalWorksize_3D[3], globalWorksize_2D[3];
    size_t globalWorksize_bf_3D[3], globalWorksize_bf_2D[3];
    size_t globalWorksize_split[2];

	enum buff_id {CVC_LIMG = 0, CVC_RIMG, CVC_LGRDX, CVC_RGRDX, CV_LCV, CV_RCV, DS_LDM, DS_RDM};

	cl_mem Ir, Ig, Ib;
	cl_mem mean_Ir, mean_Ig, mean_Ib;
	cl_mem mean_cv;
//	cl_mem Icv_r, Icv_g, Icv_b;
//	cl_mem mean_Icv_r, mean_Icv_g, mean_Icv_b;
//	cl_mem mean_Ir_cv, mean_Ig_cv, mean_Ib_cv;
//	cl_mem cov_Icv_r, cov_Icv_g, cov_Icv_b;
	cl_mem Irr, Irg, Irb, Igg, Igb, Ibb;
	cl_mem mean_Irr, mean_Irg, mean_Irb, mean_Igg, mean_Igb, mean_Ibb;
	cl_mem var_Irr, var_Irg, var_Irb, var_Igg, var_Igb, var_Ibb;
	cl_mem inv_rr, inv_rg, inv_rb, inv_gg, inv_gb, inv_bb;
	cl_mem inv_ggbb, inv_gbrb, inv_rggb, inv_rrbb, inv_rbrg, inv_rrgg;
	cl_mem inv_gbgb, inv_rgbb, inv_ggrb, inv_rbrb, inv_rrgb, inv_rgrg;
	cl_mem covDet;
//	cl_mem a_r, a_rrr, a_rgg, a_rbb;
//	cl_mem a_g, a_rrg, a_ggg, a_gbb;
//	cl_mem a_b, a_rbr, a_gbg, a_bbb;
//	cl_mem mean_Ir_ar, mean_Ig_ag, mean_Ib_ab;
//	cl_mem mean_ar, mean_ag, mean_ab;

	cl_mem b, mean_b;
	cl_mem a_r, a_g, a_b;
	cl_mem tmp_3DA_r, tmp_3DA_g, tmp_3DA_b;
	cl_mem tmp_3DB_r, tmp_3DB_g, tmp_3DB_b;

	CVF_cl(cl_context* context, cl_command_queue* commandQueue, cl_device_id device, Mat I, const int d);
	~CVF_cl(void);

	int filterCV(cl_mem* cl_Img, cl_mem* cl_costVol);

	int elementwiseMulSD(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out, size_t *globalworksize);
	int elementwiseMulDD(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out);
	int elementwiseDivSD(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out, size_t *globalworksize);
    int split(cl_mem* cl_Img, cl_mem* cl_Ir, cl_mem* cl_Ig, cl_mem* cl_Ib);
    int sub(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out, size_t *globalworksize);
    int add(cl_mem *cl_in_a, cl_mem *cl_in_b, cl_mem *cl_out, size_t *globalworksize);
    int add_constant(cl_mem *cl_in_a, float const_add, cl_mem *cl_out);
    int boxfilter(cl_mem *cl_in, cl_mem *cl_out, size_t *globalworksize);
};
