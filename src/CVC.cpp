/*---------------------------------------------------------------------------
   CVC.cpp - Cost Volume Construction Code
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
  ---------------------------------------------------------------------------*/
#include "CVC.h"

CVC::CVC(void)
{
#ifdef DEBUG_APP
		std::cout <<  "Difference of Colours and Gradients method for Cost Computation." << std::endl;
#endif // DEBUG_APP
}
CVC::~CVC(void) {}

float myCostGrd(float* lC, float* rC, float* lG, float* rG)
{
    // three color diff
	float clrDiff = fabs(lC[0] - rC[0]) + fabs(lC[1] - rC[1]) + fabs(lC[2] - rC[2]);
    // gradient diff
    float grdDiff = fabs(*lG - *rG);
    //clrDiff = clrDiff > TAU_1_32F ? TAU_1_32F : clrDiff;    //TAU_1 0.028
    //grdDiff = grdDiff > TAU_2_32F ? TAU_2_32F : grdDiff;    // TAU_2 0.008
    return ALPHA_32F * clrDiff + (1 - ALPHA_32F) * grdDiff;   // ALPHA 0.9
}

// special handle for border region
float myCostGrd(float* lC, float* lG)
{
    // three color diff
    float clrDiff = fabs(lC[0] - BC_32F) + fabs(lC[1] - BC_32F) + fabs(lC[2] - BC_32F);
    // gradient diff
    float grdDiff = fabs(*lG - BC_32F);
    //clrDiff = clrDiff > TAU_1_32F ? TAU_1_32F : clrDiff;
    //grdDiff = grdDiff > TAU_2_32F ? TAU_2_32F : grdDiff;
    return ALPHA_32F * clrDiff + (1 - ALPHA_32F) * grdDiff;
}

void CVC::preprocess(const Mat& Img, Mat& GrdX)
{
	cv::cvtColor(Img, GrdX, CV_RGB2GRAY);
	cv::Sobel(GrdX, GrdX, CV_32F, 1, 0, 1);
	return;
}

void *CVC::buildCV_left_thread(void *thread_arg)
{
    struct buildCV_TD *t_data = static_cast<struct buildCV_TD *>(thread_arg);
    Mat *lImg = t_data->lImg;
    Mat *rImg = t_data->rImg;
    Mat *lGrdX = t_data->lGrdX;
    Mat *rGrdX = t_data->rGrdX;
    const int d = t_data->d;
    Mat* costVol = t_data->costVol;

	int hei = t_data->lImg->rows;
	int wid = t_data->lImg->cols;

	for( int y = 0; y < hei; ++y) {

		float* lData = ( float* ) lImg->ptr<float>( y );
		float* rData = ( float* ) rImg->ptr<float>( y );
		float* lGData = ( float* ) lGrdX->ptr<float>( y );
		float* rGData = ( float* ) rGrdX->ptr<float>( y );
		float* cost   = ( float* ) costVol->ptr<float>( y );
		for( int x = d; x < wid; ++x) {
			float* lC = lData + 3 * x;
			float* rC = rData + 3 * ( x - d );
			float* lG = lGData + x;
			float* rG = rGData + x - d;
			cost[x] = myCostGrd( lC, rC, lG, rG );
		}
		for( int x = 0; x < d; ++x) {
			float* lC = lData + 3 * x;
			float* lG = lGData + x;
			cost[x] = myCostGrd( lC, lG );
		}
	}
    return (void*)0;

}

void *CVC::buildCV_right_thread(void *thread_arg)
{
    struct buildCV_TD *t_data;
    t_data = (struct buildCV_TD *) thread_arg;
    const Mat *lImg = t_data->lImg;
    const Mat *rImg = t_data->rImg;
    const Mat *lGrdX = t_data->lGrdX;
    const Mat *rGrdX = t_data->rGrdX;
    const int d = t_data->d;
    Mat* costVol = t_data->costVol;

	int hei = lImg->rows;
	int wid = lImg->cols;

	for( int y = 0; y < hei; ++y) {
		float* lData = ( float* ) lImg->ptr<float>( y );
		float* rData = ( float* ) rImg->ptr<float>( y );
		float* lGData = ( float* ) lGrdX->ptr<float>( y );
		float* rGData = ( float* ) rGrdX->ptr<float>( y );
		float* cost   = ( float* ) costVol->ptr<float>( y );
		for( int x = 0; x < wid - d; ++x) {
			float* lC = lData + 3 * x;
			float* rC = rData + 3 * ( x + d );
			float* lG = lGData + x;
			float* rG = rGData + x + d;
			cost[x] = myCostGrd( lC, rC, lG, rG );
		}
		for( int x = wid - d; x < wid; ++x) {
			float* lC = lData + 3 * x;
			float* lG = lGData + x;
			cost[x] = myCostGrd( lC, lG );
		}
	}

    return (void*)0;
}

void CVC::buildCV_left(const Mat& lImg, const Mat& rImg, const Mat& lGrdX, const Mat& rGrdX, const int d, Mat& costVol)
{
	int wid = lImg.cols;
	int hei = lImg.rows;

	for(int y = 0; y < hei; ++y)
	{
		float* lData = (float*)lImg.ptr<float>(y);
		float* rData = (float*)rImg.ptr<float>(y);
		float* lGData = (float*)lGrdX.ptr<float>(y);
		float* rGData = (float*)rGrdX.ptr<float>(y);
		float* cost = (float*)costVol.ptr<float>(y);

		for(int x = d; x < wid; ++x) {
			float* lC = lData + 3 * x;
			float* rC = rData + 3 * (x - d);
			float* lG = lGData + x;
			float* rG = rGData + x - d;
			cost[x] = myCostGrd(lC, rC, lG, rG);
		}
		for(int x = 0; x < d; ++x) {
			float* lC = lData + 3 * x;
			float* lG = lGData + x;
			cost[x] = myCostGrd(lC, lG);
		}
	}
}

void CVC::buildCV_right(const Mat& lImg, const Mat& rImg, const Mat& lGrdX, const Mat& rGrdX, const int d, Mat& costVol)
{
	int wid = lImg.cols;
	int border = wid - d;
	int hei = lImg.rows;

	for(int y = 0; y < lImg.rows; ++y)
	{
		float* lData = (float*)lImg.ptr<float>(y);
		float* rData = (float*)rImg.ptr<float>(y);
		float* lGData = (float*)lGrdX.ptr<float>(y);
		float* rGData = (float*)rGrdX.ptr<float>(y);
		float* cost = (float*)costVol.ptr<float>(y);

		for(int x = 0; x < border; ++x) {
			float* lC = lData + 3 * x;
			float* rC = rData + 3 * (x + d);
			float* lG = lGData + x;
			float* rG = rGData + x + d;
			cost[x] = myCostGrd(lC, rC, lG, rG);
		}
		for(int x = border; x < wid; ++x) {
			float* lC = lData + 3 * x;
			float* lG = lGData + x;
			cost[x] = myCostGrd(lC, lG);
		}
	}
}
