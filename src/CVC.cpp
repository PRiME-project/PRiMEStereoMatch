/*---------------------------------------------------------------------------
   CVC.cpp - Cost Volume Construction Code
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "CVC.h"

CVC::CVC(void)
{
    //fprintf(stderr, "OpenCV Colours and Gradients method for Cost Computation\n" );
}
CVC::~CVC(void) {}

//inline uchar myCostGrd( uchar* lC, uchar* rC, uchar* lG, uchar* rG )
//{
//    ushort clrDiff = 0;
//    // three color
//    for( int c = 0; c < 3; c ++ )
//    {
//        ushort temp = abs( lC[ c ] - rC[ c ] );
//        clrDiff += temp;
//    }
//
//    clrDiff /= 3;
//    // gradient diff
//    ushort grdDiff = abs( lG[ 0 ] - rG[ 0 ] );
//    clrDiff = clrDiff > TAU_1_16U ? TAU_1_16U : clrDiff;
//    grdDiff = grdDiff > TAU_2_16U ? TAU_2_16U : grdDiff;
//    return ALPHA_16U * clrDiff + ( 1 - ALPHA_16U ) * grdDiff;
//}
//inline uchar myCostGrd( uchar* lC, uchar* lG )
//{
//    ushort clrDiff = 0;
//    // three color
//    for( int c = 0; c < 3; c ++ )
//    {
//        ushort temp = abs( lC[ c ] - BORDER_CONSTANT_8U);
//        clrDiff += temp;
//    }
//    clrDiff /= 3;
//    // gradient diff
//    ushort grdDiff = abs( lG[ 0 ] - BORDER_CONSTANT_8U );
//    clrDiff = clrDiff > TAU_1_16U ? TAU_1_16U : clrDiff;
//    grdDiff = grdDiff > TAU_2_16U ? TAU_2_16U : grdDiff;
//    return ALPHA_16U * clrDiff + ( 1 - ALPHA_16U ) * grdDiff;
//}

inline float myCostGrd( float* lC, float* rC, float* lG, float* rG )
{
    float clrDiff = 0;
    // three color
    for( int c = 0; c < 3; c ++ )
    {
        float temp = fabs( lC[ c ] - rC[ c ] );
        clrDiff += temp;
    }

    clrDiff *= 0.3333333333;
    // gradient diff
    float grdDiff = fabs( lG[ 0 ] - rG[ 0 ] );
    clrDiff = clrDiff > TAU_1_32F ? TAU_1_32F : clrDiff;    //TAU_1 0.028
    grdDiff = grdDiff > TAU_2_32F ? TAU_2_32F : grdDiff;    // TAU_2 0.008
    return ALPHA_32F * clrDiff + ( 1 - ALPHA_32F ) * grdDiff;   // ALPHA 0.9
}

// special handle for border region
inline float myCostGrd( float* lC, float* lG )
{
    float clrDiff = 0;
    // three color
    for( int c = 0; c < 3; c ++ )
    {
        float temp = fabs( lC[ c ] - BORDER_CONSTANT_32F);
        clrDiff += temp;
    }
    clrDiff *= 0.3333333333;
    // gradient diff
    float grdDiff = fabs( lG[ 0 ] - BORDER_CONSTANT_32F );
    clrDiff = clrDiff > TAU_1_32F ? TAU_1_32F : clrDiff;
    grdDiff = grdDiff > TAU_2_32F ? TAU_2_32F : grdDiff;
    return ALPHA_32F * clrDiff + ( 1 - ALPHA_32F ) * grdDiff;
}

void CVC::preprocess(const Mat& Img, Mat& GrdX)
{
	Mat Gray;
	cvtColor(Img, Gray, CV_RGB2GRAY );

	//Sobel filter to compute X gradient
	//32 bit float data
//	if(Img.type() == CV_32FC3)
//	{
		Sobel(Gray, GrdX, CV_32F, 1, 0, 1 );
		GrdX += 0.5;
//	}
//    //8 bit unsigned char data
//    else if(Img.type() == CV_8UC3)
//    {
//		Sobel(Gray, GrdX, CV_8U, 1, 0, 1);
//		//GrdX += 128;
//    }
//    else{
//		printf("CVC: Error - Unrecognised data type in processing! (preprocess)\n");
//		exit(1);
//    }
	return;
}

void CVC::buildCV_left(const Mat& lImg, const Mat& rImg, const Mat& lGrdX, const Mat& rGrdX, const int d, Mat& costVol)
{
	int hei = lImg.rows;
	int wid = lImg.cols;

    //32 bit float data
//	if(lImg.type() == CV_32FC3 && rImg.type() == CV_32FC3)
//	{
//		CV_Assert( lImg.type() == CV_32FC3 && rImg.type() == CV_32FC3 );
		for( int y = 0; y < hei; y ++ ) {
			float* lData = ( float* ) lImg.ptr<float>( y );
			float* rData = ( float* ) rImg.ptr<float>( y );
			float* lGData = ( float* ) lGrdX.ptr<float>( y );
			float* rGData = ( float* ) rGrdX.ptr<float>( y );
			float* cost   = ( float* ) costVol.ptr<float>( y );
			for( int x = d; x < wid; x ++ ) {
				float* lC = lData + 3 * x;
				float* rC = rData + 3 * ( x - d );
				float* lG = lGData + x;
				float* rG = rGData + x - d;
				cost[x] = myCostGrd( lC, rC, lG, rG );
			}
			for( int x = 0; x < d; x ++ ) {
				float* lC = lData + 3 * x;
				float* lG = lGData + x;
				cost[x] = myCostGrd( lC, lG );
			}
		}
//    }
//    //8 bit unsigned char data
//    else if(lImg.type() == CV_8UC3 && rImg.type() == CV_8UC3)
//    {
//		CV_Assert( lImg.type() == CV_8UC3 && rImg.type() == CV_8UC3 );
//		for( int y = 0; y < hei; y ++ ) {
//			uchar* lData = ( uchar* ) lImg.ptr<uchar>( y );
//			uchar* rData = ( uchar* ) rImg.ptr<uchar>( y );
//			uchar* lGData = ( uchar* ) lGrdX.ptr<uchar>( y );
//			uchar* rGData = ( uchar* ) rGrdX.ptr<uchar>( y );
//			uchar* cost   = ( uchar* ) costVol.ptr<uchar>( y );
//			for( int x = d; x < wid; x ++ ) {
//				uchar* lC = lData + 3 * x;
//				uchar* rC = rData + 3 * ( x - d );
//				uchar* lG = lGData + x;
//				uchar* rG = rGData + x - d;
//				cost[x] = myCostGrd( lC, rC, lG, rG );
//			}
//			for( int x = 0; x < d; x ++ ) {
//				uchar* lC = lData + 3 * x;
//				uchar* lG = lGData + x;
//				cost[x] = myCostGrd( lC, lG );
//			}
//		}
//    }
//    else{
//		printf("CVC: Error - Unrecognised data type in processing! (buildCV_left)\n");
//		exit(1);
//    }
}

void CVC::buildCV_right(const Mat& lImg, const Mat& rImg, const Mat& lGrdX, const Mat& rGrdX, const int d, Mat& costVol)
{
	int hei = lImg.rows;
	int wid = lImg.cols;

    //32 bit float data
//	if(lImg.type() == CV_32FC3 && rImg.type() == CV_32FC3)
//	{
		for( int y = 0; y < hei; y ++ ) {
			float* lData = ( float* ) lImg.ptr<float>( y );
			float* rData = ( float* ) rImg.ptr<float>( y );
			float* lGData = ( float* ) lGrdX.ptr<float>( y );
			float* rGData = ( float* ) rGrdX.ptr<float>( y );
			float* cost   = ( float* ) costVol.ptr<float>( y );
			for( int x = 0; x < wid - d; x ++ ) {
				float* lC = lData + 3 * x;
				float* rC = rData + 3 * ( x + d );
				float* lG = lGData + x;
				float* rG = rGData + x + d;
				cost[x] = myCostGrd( lC, rC, lG, rG );
			}
			for( int x = wid - d; x < wid; x ++ ) {
				float* lC = lData + 3 * x;
				float* lG = lGData + x;
				cost[x] = myCostGrd( lC, lG );
			}
		}
//    }
//    //8 bit unsigned char data
//    else if(lImg.type() == CV_8UC3 && rImg.type() == CV_8UC3)
//    {
//		for( int y = 0; y < hei; y ++ ) {
//			uchar* lData = ( uchar* ) lImg.ptr<uchar>( y );
//			uchar* rData = ( uchar* ) rImg.ptr<uchar>( y );
//			uchar* lGData = ( uchar* ) lGrdX.ptr<uchar>( y );
//			uchar* rGData = ( uchar* ) rGrdX.ptr<uchar>( y );
//			uchar* cost   = ( uchar* ) costVol.ptr<uchar>( y );
//			for( int x = 0; x < wid - d; x ++ ) {
//				uchar* lC = lData + 3 * x;
//				uchar* rC = rData + 3 * ( x + d );
//				uchar* lG = lGData + x;
//				uchar* rG = rGData + x + d;
//				cost[x] = myCostGrd( lC, rC, lG, rG );
//			}
//			for( int x = wid - d; x < wid; x ++ ) {
//				uchar* lC = lData + 3 * x;
//				uchar* lG = lGData + x;
//				cost[x] = myCostGrd( lC, lG );
//			}
//		}
//    }
//    else{
//		printf("CVC: Error - Unrecognised data type in processing! (buildCV_right)\n");
//		exit(1);
//    }
}

void *CVC::buildCV_left_thread(void *thread_arg)
{
    struct buildCV_TD *t_data;
    t_data = (struct buildCV_TD *) thread_arg;
    const Mat lImg = *t_data->lImg;
    const Mat rImg = *t_data->rImg;
    const Mat lGrdX = *t_data->lGrdX;
    const Mat rGrdX = *t_data->rGrdX;
    const int d = t_data->d;
    Mat* costVol = t_data->costVol;

	int hei = lImg.rows;
	int wid = lImg.cols;

    //32 bit float data
//	if(lImg.type() == CV_32FC3 && rImg.type() == CV_32FC3)
//	{
		for( int y = 0; y < hei; y ++ ) {
			float* lData = ( float* ) lImg.ptr<float>( y );
			float* rData = ( float* ) rImg.ptr<float>( y );
			float* lGData = ( float* ) lGrdX.ptr<float>( y );
			float* rGData = ( float* ) rGrdX.ptr<float>( y );
			float* cost   = ( float* ) costVol->ptr<float>( y );
			for( int x = d; x < wid; x ++ ) {
				float* lC = lData + 3 * x;
				float* rC = rData + 3 * ( x - d );
				float* lG = lGData + x;
				float* rG = rGData + x - d;
				cost[x] = myCostGrd( lC, rC, lG, rG );
			}
			for( int x = 0; x < d; x ++ ) {
				float* lC = lData + 3 * x;
				float* lG = lGData + x;
				cost[x] = myCostGrd( lC, lG );
			}
		}
//    }
//    //8 bit unsigned char data
//    else if(lImg.type() == CV_8UC3 && rImg.type() == CV_8UC3)
//    {
//		for( int y = 0; y < hei; y ++ ) {
//			uchar* lData = ( uchar* ) lImg.ptr<uchar>( y );
//			uchar* rData = ( uchar* ) rImg.ptr<uchar>( y );
//			uchar* lGData = ( uchar* ) lGrdX.ptr<uchar>( y );
//			uchar* rGData = ( uchar* ) rGrdX.ptr<uchar>( y );
//			uchar* cost   = ( uchar* ) costVol->ptr<uchar>( y );
//			for( int x = d; x < wid; x ++ ) {
//				uchar* lC = lData + 3 * x;
//				uchar* rC = rData + 3 * ( x - d );
//				uchar* lG = lGData + x;
//				uchar* rG = rGData + x - d;
//				cost[x] = myCostGrd( lC, rC, lG, rG );
//			}
//			for( int x = 0; x < d; x ++ ) {
//				uchar* lC = lData + 3 * x;
//				uchar* lG = lGData + x;
//				cost[x] = myCostGrd( lC, lG );
//			}
//		}
//    }
//    else{
//		printf("CVC: Error - Unrecognised data type in processing! (buildCV_left_thread)\n");
//		exit(1);
//    }
    return (void*)0;
}

void *CVC::buildCV_right_thread(void *thread_arg)
{
    struct buildCV_TD *t_data;
    t_data = (struct buildCV_TD *) thread_arg;
    const Mat lImg = *t_data->lImg;
    const Mat rImg = *t_data->rImg;
    const Mat lGrdX = *t_data->lGrdX;
    const Mat rGrdX = *t_data->rGrdX;
    const int d = t_data->d;
    Mat* costVol = t_data->costVol;

	int hei = lImg.rows;
	int wid = lImg.cols;

    //32 bit float data
//	if(lImg.type() == CV_32FC3 && rImg.type() == CV_32FC3)
//	{
		for( int y = 0; y < hei; y ++ ) {
			float* lData = ( float* ) lImg.ptr<float>( y );
			float* rData = ( float* ) rImg.ptr<float>( y );
			float* lGData = ( float* ) lGrdX.ptr<float>( y );
			float* rGData = ( float* ) rGrdX.ptr<float>( y );
			float* cost   = ( float* ) costVol->ptr<float>( y );
			for( int x = 0; x < wid - d; x ++ ) {
				float* lC = lData + 3 * x;
				float* rC = rData + 3 * ( x + d );
				float* lG = lGData + x;
				float* rG = rGData + x + d;
				cost[x] = myCostGrd( lC, rC, lG, rG );
			}
			for( int x = wid - d; x < wid; x ++ ) {
				float* lC = lData + 3 * x;
				float* lG = lGData + x;
				cost[x] = myCostGrd( lC, lG );
			}
		}
//    }
//    //8 bit unsigned char data
//    else if(lImg.type() == CV_8UC3 && rImg.type() == CV_8UC3)
//    {
//		for( int y = 0; y < hei; y ++ ) {
//			uchar* lData = ( uchar* ) lImg.ptr<uchar>( y );
//			uchar* rData = ( uchar* ) rImg.ptr<uchar>( y );
//			uchar* lGData = ( uchar* ) lGrdX.ptr<uchar>( y );
//			uchar* rGData = ( uchar* ) rGrdX.ptr<uchar>( y );
//			uchar* cost   = ( uchar* ) costVol->ptr<uchar>( y );
//			for( int x = 0; x < wid; x ++ ) {
//				if( x + d < wid ) {
//					uchar* lC = lData + 3 * x;
//					uchar* rC = rData + 3 * ( x + d );
//					uchar* lG = lGData + x;
//					uchar* rG = rGData + x + d;
//					cost[x] = myCostGrd( lC, rC, lG, rG );
//				} else {
//					uchar* lC = lData + 3 * x;
//					uchar* lG = lGData + x;
//					cost[x] = myCostGrd( lC, lG );
//				}
//
//			}
//		}
//    }
//    else{
//		printf("CVC: Error - Unrecognised data type in processing! (buildCV_right_thread)\n");
//		exit(1);
//    }
    return (void*)0;
}
