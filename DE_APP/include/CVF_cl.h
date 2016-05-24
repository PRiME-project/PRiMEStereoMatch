/*---------------------------------------------------------------------------
   CVF_cl.h - OpenCL Cost Volume Filter Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "BoxFilter.h"
#include "common.h"
#include "image.h"

#define R_WIN 9
#define EPS 0.0001

//
// GIF for Cost Computation
//
class CVF_cl
{
public:

    Mat Img_ref;
	int hei;
    int wid;
    int maxDis;

    BoxFilter *bf_2D, *bf_3D;

	//Mat are 2D (images) objects
	Mat rgb[3];
//	Mat mean_I_r, mean_I_g, mean_I_b;
//	Mat rgb_rr, rgb_rg, rgb_rb, rgb_gg, rgb_gb, rgb_bb;
//	Mat rgb_rr_bf, rgb_rg_bf, rgb_rb_bf, rgb_gg_bf, rgb_gb_bf, rgb_bb_bf;
//	Mat var_I_rr, var_I_rg, var_I_rb, var_I_gg, var_I_gb, var_I_bb;
//	Mat invrr, invrg, invrb, invgg, invgb, invbb;
//	Mat covDet;
	Mat tmp_2DA_r, tmp_2DA_g, tmp_2DA_b;
	Mat tmp_2DA_rr, tmp_2DA_rg, tmp_2DA_rb;
	Mat tmp_2DA_gg, tmp_2DA_gb, tmp_2DA_bb;
	Mat tmp_2DB_rr, tmp_2DB_rg, tmp_2DB_rb;
	Mat tmp_2DB_gg, tmp_2DB_gb, tmp_2DB_bb;

	//Mat* are 3D (volume) objects
	Mat* mean_cv;
//	Mat *Icv_r, *Icv_g, *Icv_b;
//	Mat *mean_Icv_r, *mean_Icv_g, *mean_Icv_b;
//	Mat *cov_Icv_r, *cov_Icv_g, *cov_Icv_b;
//	Mat *a_r, *a_g, *a_b;
//	Mat *a_r_bf, *a_g_bf, *a_b_bf;
	Mat* tmp_3DA_r, *tmp_3DA_g, *tmp_3DA_b;
	Mat* tmp_3DB_r, *tmp_3DB_g, *tmp_3DB_b;
	Mat *b, *b_bf;

	CVF_cl(Mat I, const int d);
	~CVF_cl(void);

	int filterCV(const Mat& Img, Mat* costVol);
	int filterCV_alpha(const Mat& Img, Mat* costVol);
};
