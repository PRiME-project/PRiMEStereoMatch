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

CVF_cl::CVF_cl(Mat I, const int d) : Img_ref(I), maxDis(d)
{
    hei = Img_ref.rows;
    wid = Img_ref.cols;

    bf_2D = new BoxFilter(hei, wid, R_WIN, 1);
    bf_3D = new BoxFilter(hei, wid, R_WIN, maxDis);

	tmp_2DA_r = Mat::zeros(hei, wid, CV_32F);
	tmp_2DA_g = Mat::zeros(hei, wid, CV_32F);
	tmp_2DA_b = Mat::zeros(hei, wid, CV_32F);
	tmp_2DA_rr = Mat::zeros(hei, wid, CV_32F);
	tmp_2DA_rg = Mat::zeros(hei, wid, CV_32F);
	tmp_2DA_rb = Mat::zeros(hei, wid, CV_32F);
	tmp_2DA_gg = Mat::zeros(hei, wid, CV_32F);
	tmp_2DA_gb = Mat::zeros(hei, wid, CV_32F);
	tmp_2DA_bb = Mat::zeros(hei, wid, CV_32F);
	tmp_2DB_rr = Mat::zeros(hei, wid, CV_32F);
	tmp_2DB_rg = Mat::zeros(hei, wid, CV_32F);
	tmp_2DB_rb = Mat::zeros(hei, wid, CV_32F);
	tmp_2DB_gg = Mat::zeros(hei, wid, CV_32F);
	tmp_2DB_gb = Mat::zeros(hei, wid, CV_32F);
	tmp_2DB_bb = Mat::zeros(hei, wid, CV_32F);

	//printf("Allocated 2D Matricies\n");

	mean_cv = new Mat[maxDis];
	b = new Mat[maxDis];
	b_bf = new Mat[maxDis];
	tmp_3DA_r = new Mat[maxDis];
	tmp_3DA_g = new Mat[maxDis];
	tmp_3DA_b = new Mat[maxDis];
	tmp_3DB_r = new Mat[maxDis];
	tmp_3DB_g = new Mat[maxDis];
	tmp_3DB_b = new Mat[maxDis];

	for (int i = 0; i < maxDis; i++)
    {
        mean_cv[i] = Mat::zeros(hei, wid, CV_32FC1);
        b[i] = Mat::zeros(hei, wid, CV_32FC1);
        b_bf[i] = Mat::zeros(hei, wid, CV_32FC1);
		tmp_3DA_r[i] = Mat::zeros(hei, wid, CV_32FC1);
		tmp_3DA_g[i] = Mat::zeros(hei, wid, CV_32FC1);
		tmp_3DA_b[i] = Mat::zeros(hei, wid, CV_32FC1);
		tmp_3DB_r[i] = Mat::zeros(hei, wid, CV_32FC1);
		tmp_3DB_g[i] = Mat::zeros(hei, wid, CV_32FC1);
		tmp_3DB_b[i] = Mat::zeros(hei, wid, CV_32FC1);
    }
    //printf("Allocated 3D Matricies\n");
}

CVF_cl::~CVF_cl(void)
{
    delete [] mean_cv;
    delete [] b;
    delete [] b_bf;
    delete [] tmp_3DA_r;
    delete [] tmp_3DA_g;
    delete [] tmp_3DA_b;
    delete [] tmp_3DB_r;
    delete [] tmp_3DB_g;
    delete [] tmp_3DB_b;
}

int CVF_cl::filterCV(const Mat& Img, Mat* costVol)
{
//    float time_start = get_rt_cvf_cl();
//    printf("0 Time from start = %.0f us\n", get_rt_cvf_cl() - time_start);
    split(Img, rgb);

//	printf("1 Time from start = %.0f us\n", get_rt_cvf_cl() - time_start);

	bf_2D->filter(&rgb[0], &tmp_2DA_r);
	bf_2D->filter(&rgb[1], &tmp_2DA_g);
	bf_2D->filter(&rgb[2], &tmp_2DA_b);

//	printf("2 Time from start = %.0f us\n", get_rt_cvf_cl() - time_start);

	bf_3D->filter(costVol, mean_cv);

//    printf("3 Time from start = %.0f us\n", get_rt_cvf_cl() - time_start);

	for (int i = 0; i < maxDis; i++)
    {
		tmp_3DA_r[i] = rgb[0].mul(costVol[i]);
		tmp_3DA_g[i] = rgb[1].mul(costVol[i]);
		tmp_3DA_b[i] = rgb[2].mul(costVol[i]);
	}
//	printf("4 Time from start = %.0f us\n", get_rt_cvf_cl() - time_start);

	bf_3D->filter(tmp_3DA_r, tmp_3DB_r);
	bf_3D->filter(tmp_3DA_g, tmp_3DB_g);
	bf_3D->filter(tmp_3DA_b, tmp_3DB_b);

//	printf("5 Time from start = %.0f us\n", get_rt_cvf_cl() - time_start);
//	exit(1);

	for (int i = 0; i < maxDis; i++)
    {
		// covariance of (I, p) in each local patch.
		tmp_3DA_r[i] = tmp_3DB_r[i] - tmp_2DA_r.mul(mean_cv[i]);
		tmp_3DA_g[i] = tmp_3DB_g[i] - tmp_2DA_g.mul(mean_cv[i]);
		tmp_3DA_b[i] = tmp_3DB_b[i] - tmp_2DA_b.mul(mean_cv[i]);
	}

    // variance of I in each local patch: the matrix Sigma in Eqn (14).
    // Note the variance in each local patch is a 3x3 symmetric matrix:
    //           rr, rg, rb
    //   Sigma = rg, gg, gb
    //           rb, gb, bb
    tmp_2DA_rr = rgb[0].mul(rgb[0]);
    tmp_2DA_rg = rgb[0].mul(rgb[1]);
    tmp_2DA_rb = rgb[0].mul(rgb[2]);
    tmp_2DA_gg = rgb[1].mul(rgb[1]);
    tmp_2DA_gb = rgb[1].mul(rgb[2]);
    tmp_2DA_bb = rgb[2].mul(rgb[2]);

	bf_2D->filter(&tmp_2DA_rr, &tmp_2DB_rr);
	bf_2D->filter(&tmp_2DA_rg, &tmp_2DB_rg);
	bf_2D->filter(&tmp_2DA_rb, &tmp_2DB_rb);
	bf_2D->filter(&tmp_2DA_gg, &tmp_2DB_gg);
	bf_2D->filter(&tmp_2DA_gb, &tmp_2DB_gb);
	bf_2D->filter(&tmp_2DA_bb, &tmp_2DB_bb);

    tmp_2DA_rr = tmp_2DB_rr - tmp_2DA_r.mul(tmp_2DA_r) + EPS;
    tmp_2DA_rg = tmp_2DB_rg - tmp_2DA_r.mul(tmp_2DA_g);
    tmp_2DA_rb = tmp_2DB_rb - tmp_2DA_r.mul(tmp_2DA_g);
    tmp_2DA_gg = tmp_2DB_gg - tmp_2DA_g.mul(tmp_2DA_g) + EPS;
    tmp_2DA_gb = tmp_2DB_gb - tmp_2DA_g.mul(tmp_2DA_b);
    tmp_2DA_bb = tmp_2DB_bb - tmp_2DA_b.mul(tmp_2DA_b) + EPS;

    // Inverse of Sigma + eps * I
    tmp_2DB_rr = tmp_2DA_gg.mul(tmp_2DA_bb) - tmp_2DA_gb.mul(tmp_2DA_gb);
    tmp_2DB_rg = tmp_2DA_gb.mul(tmp_2DA_rb) - tmp_2DA_rg.mul(tmp_2DA_bb);
    tmp_2DB_rb = tmp_2DA_rg.mul(tmp_2DA_gb) - tmp_2DA_gg.mul(tmp_2DA_rb);
    tmp_2DB_gg = tmp_2DA_rr.mul(tmp_2DA_bb) - tmp_2DA_rb.mul(tmp_2DA_rb);
    tmp_2DB_gb = tmp_2DA_rb.mul(tmp_2DA_rg) - tmp_2DA_rr.mul(tmp_2DA_gb);
    tmp_2DB_bb = tmp_2DA_rr.mul(tmp_2DA_gg) - tmp_2DA_rg.mul(tmp_2DA_rg);

    tmp_2DA_rr = tmp_2DB_rr.mul(tmp_2DA_rr) + tmp_2DB_rg.mul(tmp_2DA_rg) + tmp_2DB_rb.mul(tmp_2DA_rb);

    tmp_2DB_rr /= tmp_2DA_rr;
    tmp_2DB_rg /= tmp_2DA_rr;
    tmp_2DB_rb /= tmp_2DA_rr;
    tmp_2DB_gg /= tmp_2DA_rr;
    tmp_2DB_gb /= tmp_2DA_rr;
    tmp_2DB_bb /= tmp_2DA_rr;

	for (int i = 0; i < maxDis; i++)
    {
		tmp_3DB_r[i] = tmp_2DB_rr.mul(tmp_3DA_r[i]) + tmp_2DB_rg.mul(tmp_3DA_g[i]) + tmp_2DB_rb.mul(tmp_3DA_b[i]);
		tmp_3DB_g[i] = tmp_2DB_rg.mul(tmp_3DA_r[i]) + tmp_2DB_gg.mul(tmp_3DA_g[i]) + tmp_2DB_gb.mul(tmp_3DA_b[i]);
		tmp_3DB_b[i] = tmp_2DB_rb.mul(tmp_3DA_r[i]) + tmp_2DB_gb.mul(tmp_3DA_g[i]) + tmp_2DB_bb.mul(tmp_3DA_b[i]);

		b[i] = mean_cv[i] - tmp_3DB_r[i].mul(tmp_2DA_r) - tmp_3DB_g[i].mul(tmp_2DA_g) - tmp_3DB_b[i].mul(tmp_2DA_b); // Eqn. (15) in the paper;
	}

	bf_3D->filter(tmp_3DB_r, tmp_3DA_r);
	bf_3D->filter(tmp_3DB_g, tmp_3DA_g);
	bf_3D->filter(tmp_3DB_b, tmp_3DA_b);
	bf_3D->filter(b, b_bf);

	for (int i = 0; i < maxDis; i++)
    {
		costVol[i] = (tmp_3DA_r[i].mul(rgb[0]) + tmp_3DA_g[i].mul(rgb[1]) + tmp_3DA_b[i].mul(rgb[2]) + b_bf[i]);  // Eqn. (16) in the paper;
	}
    return 0;
}
//int CVF_cl::filterCV_alpha(const Mat& Img, Mat* costVol)
//{
//    split(Img, rgb);
//
//	bf_2D->filter(&rgb[0], &mean_I_r);
//	bf_2D->filter(&rgb[1], &mean_I_g);
//	bf_2D->filter(&rgb[2], &mean_I_b);
//
//	bf_3D->filter(costVol, mean_cv);
//
//	for (int i = 0; i < maxDis; i++)
//    {
//		Icv_r[i] = rgb[0].mul(costVol[i]);
//		Icv_g[i] = rgb[1].mul(costVol[i]);
//		Icv_b[i] = rgb[2].mul(costVol[i]);
//	}
//
//	bf_3D->filter(Icv_r, mean_Icv_r);
//	bf_3D->filter(Icv_g, mean_Icv_g);
//	bf_3D->filter(Icv_b, mean_Icv_b);
//
//	for (int i = 0; i < maxDis; i++)
//    {
//		// covariance of (I, p) in each local patch.
//		cov_Icv_r[i] = mean_Icv_r[i] - mean_I_r.mul(mean_cv[i]);
//		cov_Icv_g[i] = mean_Icv_g[i] - mean_I_g.mul(mean_cv[i]);
//		cov_Icv_b[i] = mean_Icv_b[i] - mean_I_b.mul(mean_cv[i]);
//	}
//
//    // variance of I in each local patch: the matrix Sigma in Eqn (14).
//    // Note the variance in each local patch is a 3x3 symmetric matrix:
//    //           rr, rg, rb
//    //   Sigma = rg, gg, gb
//    //           rb, gb, bb
//    rgb_rr = rgb[0].mul(rgb[0]);
//    rgb_rg = rgb[0].mul(rgb[1]);
//    rgb_rb = rgb[0].mul(rgb[2]);
//    rgb_gg = rgb[1].mul(rgb[1]);
//    rgb_gb = rgb[1].mul(rgb[2]);
//    rgb_bb = rgb[2].mul(rgb[2]);
//
//	bf_2D->filter(&rgb_rr, &rgb_rr_bf);
//	bf_2D->filter(&rgb_rg, &rgb_rg_bf);
//	bf_2D->filter(&rgb_rb, &rgb_rb_bf);
//	bf_2D->filter(&rgb_gg, &rgb_gg_bf);
//	bf_2D->filter(&rgb_gb, &rgb_gb_bf);
//	bf_2D->filter(&rgb_bb, &rgb_bb_bf);
//
//    var_I_rr = rgb_rr_bf - mean_I_r.mul(mean_I_r) + EPS;
//    var_I_rg = rgb_rg_bf - mean_I_r.mul(mean_I_g);
//    var_I_rb = rgb_rb_bf - mean_I_r.mul(mean_I_b);
//    var_I_gg = rgb_gg_bf - mean_I_g.mul(mean_I_g) + EPS;
//    var_I_gb = rgb_gb_bf - mean_I_g.mul(mean_I_b);
//    var_I_bb = rgb_bb_bf - mean_I_b.mul(mean_I_b) + EPS;
//
//    // Inverse of Sigma + eps * I
//    invrr = var_I_gg.mul(var_I_bb) - var_I_gb.mul(var_I_gb);
//    invrg = var_I_gb.mul(var_I_rb) - var_I_rg.mul(var_I_bb);
//    invrb = var_I_rg.mul(var_I_gb) - var_I_gg.mul(var_I_rb);
//    invgg = var_I_rr.mul(var_I_bb) - var_I_rb.mul(var_I_rb);
//    invgb = var_I_rb.mul(var_I_rg) - var_I_rr.mul(var_I_gb);
//    invbb = var_I_rr.mul(var_I_gg) - var_I_rg.mul(var_I_rg);
//
//    covDet = invrr.mul(var_I_rr) + invrg.mul(var_I_rg) + invrb.mul(var_I_rb);
//
//    invrr /= covDet;
//    invrg /= covDet;
//    invrb /= covDet;
//    invgg /= covDet;
//    invgb /= covDet;
//    invbb /= covDet;
//
//	for (int i = 0; i < maxDis; i++)
//    {
//		a_r[i] = invrr.mul(cov_Icv_r[i]) + invrg.mul(cov_Icv_g[i]) + invrb.mul(cov_Icv_b[i]);
//		a_g[i] = invrg.mul(cov_Icv_r[i]) + invgg.mul(cov_Icv_g[i]) + invgb.mul(cov_Icv_b[i]);
//		a_b[i] = invrb.mul(cov_Icv_r[i]) + invgb.mul(cov_Icv_g[i]) + invbb.mul(cov_Icv_b[i]);
//
//		b[i] = mean_cv[i] - a_r[i].mul(mean_I_r) - a_g[i].mul(mean_I_g) - a_b[i].mul(mean_I_b); // Eqn. (15) in the paper;
//	}
//
//	bf_3D->filter(a_r, a_r_bf);
//	bf_3D->filter(a_g, a_g_bf);
//	bf_3D->filter(a_b, a_b_bf);
//	bf_3D->filter(b, b_bf);
//
//	for (int i = 0; i < maxDis; i++)
//    {
//		costVol[i] = (a_r_bf[i].mul(rgb[0]) + a_g_bf[i].mul(rgb[1]) + a_b_bf[i].mul(rgb[2]) + b_bf[i]);  // Eqn. (16) in the paper;
//	}
//    return 0;
//}
