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

inline double myCostGrd( double* lC, double* rC, double* lG, double* rG )
{
    double clrDiff = 0;
    // three color
    for( int c = 0; c < 3; c ++ )
    {
        double temp = fabs( lC[ c ] - rC[ c ] );
        clrDiff += temp;
    }

    clrDiff *= 0.3333333333;
    // gradient diff
    double grdDiff = fabs( lG[ 0 ] - rG[ 0 ] );
    clrDiff = clrDiff > TAU_1 ? TAU_1 : clrDiff;    //TAU_1 0.028
    grdDiff = grdDiff > TAU_2 ? TAU_2 : grdDiff;    // TAU_2 0.008
    return ALPHA * clrDiff + ( 1 - ALPHA ) * grdDiff;   // ALPHA 0.9
}
// special handle for border region
inline double myCostGrd( double* lC, double* lG )
{
    double clrDiff = 0;
    // three color
    for( int c = 0; c < 3; c ++ )
    {
        double temp = fabs( lC[ c ] - BORDER_CONSTANT);
        clrDiff += temp;
    }
    clrDiff *= 0.3333333333;
    // gradient diff
    double grdDiff = fabs( lG[ 0 ] - BORDER_CONSTANT );
    clrDiff = clrDiff > TAU_1 ? TAU_1 : clrDiff;
    grdDiff = grdDiff > TAU_2 ? TAU_2 : grdDiff;
    return ALPHA * clrDiff + ( 1 - ALPHA ) * grdDiff;
}

void CVC::buildCV(const Mat& lImg, const Mat& rImg, const int d, Mat& costVol)
{
	CV_Assert( lImg.type() == CV_64FC3 && rImg.type() == CV_64FC3 );

	int hei = lImg.rows;
	int wid = lImg.cols;
	Mat lGray, rGray;
	Mat lGrdX, rGrdX;
	Mat tmp;
	lImg.convertTo( tmp, CV_32F );
	cvtColor(tmp, lGray, CV_RGB2GRAY );
	rImg.convertTo( tmp, CV_32F );
	cvtColor( tmp, rGray, CV_RGB2GRAY );

    //Sobel filter to compute X gradient     <-- investigate Mali Sobel OpenCL kernel
    Sobel( lGray, lGrdX, CV_64F, 1, 0, 1 );
	Sobel( rGray, rGrdX, CV_64F, 1, 0, 1 );
	lGrdX += 0.5;
	rGrdX += 0.5;

    for( int y = 0; y < hei; y ++ ) {
        double* lData = ( double* ) lImg.ptr<double>( y );
        double* rData = ( double* ) rImg.ptr<double>( y );
        double* lGData = ( double* ) lGrdX.ptr<double>( y );
        double* rGData = ( double* ) rGrdX.ptr<double>( y );
        double* cost   = ( double* ) costVol.ptr<double>( y );
        for( int x = 0; x < wid; x ++ ) {
            if( x - abs(d) >= 0 ) {
                double* lC = lData + 3 * x;
                double* rC = rData + 3 * ( x - d );
                double* lG = lGData + x;
                double* rG = rGData + x - d;
                cost[x] = myCostGrd( lC, rC, lG, rG );
            } else {
                double* lC = lData + 3 * x;
                double* lG = lGData + x;
                cost[x] = myCostGrd( lC, lG );
            }

        }
    }
}

void *CVC::buildCV_thread(void *thread_arg)
{
    struct buildCV_TD *t_data;
    t_data = (struct buildCV_TD *) thread_arg;
    const Mat lImg = *t_data->lImg;
    const Mat rImg = *t_data->rImg;
    const int d = t_data->d;
    Mat* costVol = t_data->costVol;

	CV_Assert( lImg.type() == CV_64FC3 && rImg.type() == CV_64FC3 );

	int hei = lImg.rows;
	int wid = lImg.cols;
	Mat lGray, rGray;
	Mat lGrdX, rGrdX;
	Mat tmp;
	lImg.convertTo( tmp, CV_32F );
	cvtColor(tmp, lGray, CV_RGB2GRAY );
	rImg.convertTo( tmp, CV_32F );
	cvtColor( tmp, rGray, CV_RGB2GRAY );

    //Sobel filter to compute X gradient     <-- investigate Mali Sobel OpenCL kernel
    Sobel( lGray, lGrdX, CV_64F, 1, 0, 1 );
	Sobel( rGray, rGrdX, CV_64F, 1, 0, 1 );
	lGrdX += 0.5;
	rGrdX += 0.5;

    for( int y = 0; y < hei; y ++ ) {
        double* lData = ( double* ) lImg.ptr<double>( y );
        double* rData = ( double* ) rImg.ptr<double>( y );
        double* lGData = ( double* ) lGrdX.ptr<double>( y );
        double* rGData = ( double* ) rGrdX.ptr<double>( y );
        double* cost   = ( double* ) costVol->ptr<double>( y );
        for( int x = 0; x < wid; x ++ ) {
            if( x - abs(d) >= 0 ) {
                double* lC = lData + 3 * x;
                double* rC = rData + 3 * ( x - d );
                double* lG = lGData + x;
                double* rG = rGData + x - d;
                cost[x] = myCostGrd( lC, rC, lG, rG );
            } else {
                double* lC = lData + 3 * x;
                double* lG = lGData + x;
                cost[x] = myCostGrd( lC, lG );
            }

        }
    }
    return (void*)0;
}

