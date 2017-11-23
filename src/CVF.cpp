/*---------------------------------------------------------------------------
   CVF.cpp - Cost Volume Filter Code
           - Guided Image Filter
           - Box Filter
  ---------------------------------------------------------------------------
   Co-Author: Charan Kumar
   Email: EE14MTECH01008@iith.ac.in
   Co-Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "CVF.h"

float get_rt_cvf(){
	struct timespec realtime;
	clock_gettime(CLOCK_MONOTONIC,&realtime);
	return (float)(realtime.tv_sec*1000000+realtime.tv_nsec/1000);
}

CVF::CVF(void)
{
    //fprintf(stderr, "OpenCV Guided Image Filtering method for Cost Computation\n" );
}
CVF::~CVF(void){}

void *CVF::filterCV_thread(void *thread_arg)
{
    struct filterCV_TD *t_data;
    t_data = (struct filterCV_TD *) thread_arg;
    const Mat* Img_rgb = t_data->Img_rgb;
    const Mat* mean_Img = t_data->mean_Img;
    const Mat* var_Img = t_data->var_Img;
    const Mat pImg = *t_data->costVol;
    Mat* costVol = t_data->costVol;

    *costVol = GuidedFilter_cv(Img_rgb, mean_Img, var_Img, pImg);

    return (void*)0;
}

//Image channel division, boxfiltering and variance calculation can all be computed once for all disparities
int CVF::preprocess(const Mat& Img, Mat* Img_rgb, Mat* mean_Img, Mat* var_Img)
{
    Size r = Size(GIF_R_WIN,GIF_R_WIN);

    split( Img, Img_rgb );

    for( int c = 0; c < 3; c ++ ) {
        boxFilter( Img_rgb[c], mean_Img[c], -1, r );
    }

    Mat tmp;
    int varIdx = 0;
    for( int c = 0; c < 3; c ++ ) {
        for( int c_p = c; c_p < 3; c_p ++ ) {
            multiply( Img_rgb[ c ], Img_rgb[ c_p ], tmp );
            boxFilter( tmp, var_Img[varIdx], -1, r );
            multiply( mean_Img[ c ], mean_Img[ c_p ], tmp );
            var_Img[ varIdx ] -= tmp;
            varIdx ++;
        }
    }
	return 0;
}


void CVF::filterCV(const Mat* Img_rgb, const Mat* mean_Img, const Mat* var_Img, Mat& costVol)
{
    costVol = GuidedFilter_cv(Img_rgb, mean_Img, var_Img, costVol);
    return;
}

Mat GuidedFilter_cv(const Mat* rgb, const Mat* mean_I, const Mat* var_I, const Mat& p)
{
    Size r = Size(GIF_R_WIN,GIF_R_WIN);

	int H = rgb[0].rows;
    int W = rgb[0].cols;
    // color guidence
    // image must in RGB format

//    Mat rgb[ 3 ];
//    split( I, rgb );
//    Mat mean_I[ 3 ];
//    for( int c = 0; c < 3; c ++ ) {
//        boxFilter( rgb[c], mean_I[c], -1, r );
//    }
    Mat mean_p;
    boxFilter( p, mean_p, -1, r );

    Mat tmp;
    Mat mean_Ip[ 3 ];
    for( int c = 0; c < 3; c ++ ) {
        multiply( rgb[ c ], p, tmp );
        boxFilter( tmp, mean_Ip[c], -1, r );
    }
    /*% covariance of (I, p) in each local patch.*/
    Mat cov_Ip[ 3 ];
    for( int c = 0; c < 3; c ++ ) {
        multiply( mean_I[ c ], mean_p, tmp );
        cov_Ip[ c ] = mean_Ip[ c ] - tmp;
    }

    //  % variance of I in each local patch: the matrix Sigma in Eqn (14).
    //	% Note the variance in each local patch is a 3x3 symmetric matrix:
    //  %           rr, rg, rb
    //	%   Sigma = rg, gg, gb
    //	%           rb, gb, bb
//    Mat var_I[ 6 ];
//    int varIdx = 0;
//    for( int c = 0; c < 3; c ++ ) {
//        for( int c_p = c; c_p < 3; c_p ++ ) {
//            multiply( rgb[ c ], rgb[ c_p ], tmp );
//            boxFilter( tmp, var_I[varIdx], -1, r );
//            multiply( mean_I[ c ], mean_I[ c_p ], tmp );
//            var_I[ varIdx ] -= tmp;
//            varIdx ++;
//        }
//    }

    Mat a[ 3 ];
    for( int c = 0; c < 3; c ++  )
    {
		a[ c ] = Mat::zeros( H, W, CV_32FC1 );
    }
	for( int y = 0; y < H; y ++ ) {
		float* vData[ 6 ];
		for( int v = 0; v < 6; v ++ ) {
			vData[ v ] = ( float* ) var_I[ v ].ptr<float>( y );
		}
		float* cData[ 3 ];
		for( int c = 0; c < 3; c ++ ) {
			cData[ c ] = ( float * ) cov_Ip[ c ].ptr<float>( y );
		}
		float* aData[ 3 ];
		for( int c = 0; c < 3; c++  ) {
			aData[ c ] = ( float* ) a[ c ].ptr<float>( y );
		}
		for( int x = 0; x < W; x ++ )
		{
			float c0 = cData[ 0 ][ x ];
			float c1 = cData[ 1 ][ x ];
			float c2 = cData[ 2 ][ x ];
			float a11 = vData[ 0 ][ x ] + GIF_EPS_32F;
			float a12 = vData[ 1 ][ x ];
			float a13 = vData[ 2 ][ x ];
			float a21 = vData[ 1 ][ x ];
			float a22 = vData[ 3 ][ x ] + GIF_EPS_32F;
			float a23 = vData[ 4 ][ x ];
			float a31 = vData[ 2 ][ x ];
			float a32 = vData[ 4 ][ x ];
			float a33 = vData[ 5 ][ x ] + GIF_EPS_32F;
			float DET = a11 * ( a33 * a22 - a32 * a23 ) -
				a21 * ( a33 * a12 - a32 * a13 ) +
				a31 * ( a23 * a12 - a22 * a13 );
			DET = 1 / DET;
			aData[ 0 ][ x ] = DET * (
				c0 * ( a33 * a22 - a32 * a23 ) +
				c1 * ( a31 * a23 - a33 * a21 ) +
				c2 * ( a32 * a21 - a31 * a22 )
				);
			aData[ 1 ][ x ] = DET * (
				c0 * ( a32 * a13 - a33 * a12 ) +
				c1 * ( a33 * a11 - a31 * a13 ) +
				c2 * ( a31 * a12 - a32 * a11 )
				);
			aData[ 2 ][ x ] = DET * (
				c0 * ( a23 * a12 - a22 * a13 ) +
				c1 * ( a21 * a13 - a23 * a11 ) +
				c2 * ( a22 * a11 - a21 * a12 )
				);
		}
	}


    //Mat b = mean_p.clone();
    for( int c = 0; c < 3; c ++ ) {
        multiply( a[ c ], mean_I[ c ], tmp );
        mean_p -= tmp;
    }

    Mat q;
    boxFilter( mean_p, q, -1, r );
    for( int c = 0; c < 3; c ++ ) {
        boxFilter( a[c], tmp, -1, r );
        multiply( tmp, rgb[ c ], tmp );
        q += tmp;
    }

    return q;
}
