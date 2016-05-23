/*---------------------------------------------------------------------------
   CVC.cpp - Cost Volume Construction Code
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "CVC.h"

CVC::CVC(void)
{
    //fprintf(stderr, "OpenCV Colours and Gradients method for Cost Computation\n" );
}
CVC::~CVC(void) {}

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
    clrDiff = clrDiff > TAU_1 ? TAU_1 : clrDiff;    //TAU_1 0.028
    grdDiff = grdDiff > TAU_2 ? TAU_2 : grdDiff;    // TAU_2 0.008
    return ALPHA * clrDiff + ( 1 - ALPHA ) * grdDiff;   // ALPHA 0.9
}
// special handle for border region
inline float myCostGrd( float* lC, float* lG )
{
    float clrDiff = 0;
    // three color
    for( int c = 0; c < 3; c ++ )
    {
        float temp = fabs( lC[ c ] - BORDER_CONSTANT);
        clrDiff += temp;
    }
    clrDiff *= 0.3333333333;
    // gradient diff
    float grdDiff = fabs( lG[ 0 ] - BORDER_CONSTANT );
    clrDiff = clrDiff > TAU_1 ? TAU_1 : clrDiff;
    grdDiff = grdDiff > TAU_2 ? TAU_2 : grdDiff;
    return ALPHA * clrDiff + ( 1 - ALPHA ) * grdDiff;
}

void CVC::buildCV_left(const Mat& lImg, const Mat& rImg, const int d, Mat& costVol)
{
	//CV_Assert( lImg.type() == CV_64FC3 && rImg.type() == CV_64FC3 );

	int hei = lImg.rows;
	int wid = lImg.cols;
	Mat lGray, rGray;
	Mat lGrdX, rGrdX;
	Mat tmp;
	//lImg.convertTo( tmp, CV_32F );
	cvtColor(lImg, lGray, CV_RGB2GRAY );
	//rImg.convertTo( tmp, CV_32F );
	cvtColor(rImg, rGray, CV_RGB2GRAY );

    //Sobel filter to compute X gradient     <-- investigate Mali Sobel OpenCL kernel
    Sobel( lGray, lGrdX, CV_32F, 1, 0, 1 );
	Sobel( rGray, rGrdX, CV_32F, 1, 0, 1 );
	lGrdX += 0.5;
	rGrdX += 0.5;

    for( int y = 0; y < hei; y ++ ) {
        float* lData = ( float* ) lImg.ptr<float>( y );
        float* rData = ( float* ) rImg.ptr<float>( y );
        float* lGData = ( float* ) lGrdX.ptr<float>( y );
        float* rGData = ( float* ) rGrdX.ptr<float>( y );
        float* cost   = ( float* ) costVol.ptr<float>( y );
        for( int x = 0; x < wid; x ++ ) {
            if( x - d >= 0 ) {
                float* lC = lData + 3 * x;
                float* rC = rData + 3 * ( x - d );
                float* lG = lGData + x;
                float* rG = rGData + x - d;
                cost[x] = myCostGrd( lC, rC, lG, rG );
            } else {
                float* lC = lData + 3 * x;
                float* lG = lGData + x;
                cost[x] = myCostGrd( lC, lG );
            }

        }
    }
}
void CVC::buildCV_right(const Mat& lImg, const Mat& rImg, const int d, Mat& costVol)
{
	//CV_Assert( lImg.type() == CV_64FC3 && rImg.type() == CV_64FC3 );

	int hei = lImg.rows;
	int wid = lImg.cols;
	Mat lGray, rGray;
	Mat lGrdX, rGrdX;
	Mat tmp;
	//lImg.convertTo( tmp, CV_32F );
	cvtColor(lImg, lGray, CV_RGB2GRAY );
	//rImg.convertTo( tmp, CV_32F );
	cvtColor(rImg, rGray, CV_RGB2GRAY );

    //Sobel filter to compute X gradient     <-- investigate Mali Sobel OpenCL kernel
    Sobel( lGray, lGrdX, CV_32F, 1, 0, 1 );
	Sobel( rGray, rGrdX, CV_32F, 1, 0, 1 );
	lGrdX += 0.5;
	rGrdX += 0.5;

    for( int y = 0; y < hei; y ++ ) {
        float* lData = ( float* ) lImg.ptr<float>( y );
        float* rData = ( float* ) rImg.ptr<float>( y );
        float* lGData = ( float* ) lGrdX.ptr<float>( y );
        float* rGData = ( float* ) rGrdX.ptr<float>( y );
        float* cost   = ( float* ) costVol.ptr<float>( y );
        for( int x = 0; x < wid; x ++ ) {
            if( x + d < wid ) {
                float* lC = lData + 3 * x;
                float* rC = rData + 3 * ( x + d );
                float* lG = lGData + x;
                float* rG = rGData + x + d;
                cost[x] = myCostGrd( lC, rC, lG, rG );
            } else {
                float* lC = lData + 3 * x;
                float* lG = lGData + x;
                cost[x] = myCostGrd( lC, lG );
            }

        }
    }
}

void *CVC::buildCV_left_thread(void *thread_arg)
{
    struct buildCV_TD *t_data;
    t_data = (struct buildCV_TD *) thread_arg;
    const Mat lImg = *t_data->lImg;
    const Mat rImg = *t_data->rImg;
    const int d = t_data->d;
    Mat* costVol = t_data->costVol;

	//CV_Assert( lImg.type() == CV_64FC3 && rImg.type() == CV_64FC3 );

	int hei = lImg.rows;
	int wid = lImg.cols;
	Mat lGray, rGray;
	Mat lGrdX, rGrdX;
	Mat tmp;
	//lImg.convertTo( tmp, CV_32F );
	cvtColor(lImg, lGray, CV_RGB2GRAY );
	//rImg.convertTo( tmp, CV_32F );
	cvtColor(rImg, rGray, CV_RGB2GRAY );

    //Sobel filter to compute X gradient     <-- investigate Mali Sobel OpenCL kernel
    Sobel( lGray, lGrdX, CV_32F, 1, 0, 1 );
	Sobel( rGray, rGrdX, CV_32F, 1, 0, 1 );
	lGrdX += 0.5;
	rGrdX += 0.5;

    for( int y = 0; y < hei; y ++ ) {
        float* lData = ( float* ) lImg.ptr<float>( y );
        float* rData = ( float* ) rImg.ptr<float>( y );
        float* lGData = ( float* ) lGrdX.ptr<float>( y );
        float* rGData = ( float* ) rGrdX.ptr<float>( y );
        float* cost   = ( float* ) costVol->ptr<float>( y );
        for( int x = 0; x < wid; x ++ ) {
            if( x - d >= 0 ) {
                float* lC = lData + 3 * x;
                float* rC = rData + 3 * ( x - d );
                float* lG = lGData + x;
                float* rG = rGData + x - d;
                cost[x] = myCostGrd( lC, rC, lG, rG );
            } else {
                float* lC = lData + 3 * x;
                float* lG = lGData + x;
                cost[x] = myCostGrd( lC, lG );
            }

        }
    }
    return (void*)0;
}

void *CVC::buildCV_right_thread(void *thread_arg)
{
    struct buildCV_TD *t_data;
    t_data = (struct buildCV_TD *) thread_arg;
    const Mat lImg = *t_data->lImg;
    const Mat rImg = *t_data->rImg;
    const int d = t_data->d;
    Mat* costVol = t_data->costVol;

	//CV_Assert( lImg.type() == CV_64FC3 && rImg.type() == CV_64FC3 );

	int hei = lImg.rows;
	int wid = lImg.cols;
	Mat lGray, rGray;
	Mat lGrdX, rGrdX;
	Mat tmp;
	//lImg.convertTo( tmp, CV_32F );
	cvtColor(lImg, lGray, CV_RGB2GRAY );
	//rImg.convertTo( tmp, CV_32F );
	cvtColor(rImg, rGray, CV_RGB2GRAY );

    //Sobel filter to compute X gradient     <-- investigate Mali Sobel OpenCL kernel
    Sobel( lGray, lGrdX, CV_32F, 1, 0, 1 );
	Sobel( rGray, rGrdX, CV_32F, 1, 0, 1 );
	lGrdX += 0.5;
	rGrdX += 0.5;

    for( int y = 0; y < hei; y ++ ) {
        float* lData = ( float* ) lImg.ptr<float>( y );
        float* rData = ( float* ) rImg.ptr<float>( y );
        float* lGData = ( float* ) lGrdX.ptr<float>( y );
        float* rGData = ( float* ) rGrdX.ptr<float>( y );
        float* cost   = ( float* ) costVol->ptr<float>( y );
        for( int x = 0; x < wid; x ++ ) {
            if( x + d < wid ) {
                float* lC = lData + 3 * x;
                float* rC = rData + 3 * ( x + d );
                float* lG = lGData + x;
                float* rG = rGData + x + d;
                cost[x] = myCostGrd( lC, rC, lG, rG );
            } else {
                float* lC = lData + 3 * x;
                float* lG = lGData + x;
                cost[x] = myCostGrd( lC, lG );
            }

        }
    }
    return (void*)0;
}
