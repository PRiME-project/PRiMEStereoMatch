/*---------------------------------------------------------------------------
   PP.cpp - Post Processing Code
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "PP.h"

PP::PP(void)
{
	//printf( "L-R Consistency Check and Weighted-Median Filter Post-Processing\n" );
}
PP::~PP(void) {}

void lrCheck(Mat& lDis, Mat& rDis, Mat& lValid, Mat& rValid)
{
    int hei = lDis.rows;
	int wid = lDis.cols;
    lValid = Scalar(0); //memset( lValid, 0, imgSize * sizeof( int ) );
    rValid = Scalar(0); //memset( rValid, 0, imgSize * sizeof( int ) );
    for( int y = 0; y < hei; y ++ ) {
		uchar* lDisData = ( uchar* ) lDis.ptr<uchar>( y );
		uchar* rDisData = ( uchar* ) rDis.ptr<uchar>( y );
		uchar* lValidData = ( uchar* ) lValid.ptr<uchar>( y );
		uchar* rValidData = ( uchar* ) rValid.ptr<uchar>( y );
        for( int x = 0; x < wid; x ++ ) {
            // check left image
            int lDep = lDisData[ x ];
            // assert( ( x - lDep ) >= 0 && ( x - lDep ) < wid );
            int rLoc = ( x - lDep + wid ) % wid;
            int rDep = rDisData[ rLoc ];
            // disparity should not be zero
            if( lDep == rDep && lDep >= 2 ) {
                lValidData[x] = 1;
            }
            // check right image
            rDep = rDisData[ x ];
            // assert( ( x + rDep ) >= 0 && ( x + rDep ) < wid );
            int lLoc = ( x + rDep + wid ) % wid;
            lDep = lDisData[ lLoc ];
            // disparity should not be zero
            if( rDep == lDep && rDep >= 2 ) {
                rValidData[x] = 1;
            }
        }
    }
    return;
}

void fillInv(Mat& lDis, Mat& rDis, Mat& lValid, Mat& rValid)
{
	int hei = lDis.rows;
	int wid = lDis.cols;

    // fill left dep
    for( int y = 0; y < hei; y ++ ) {
		uchar* lDisData = ( uchar* ) lDis.ptr<uchar>( y );
		uchar* lValidData = ( uchar* ) lValid.ptr<uchar>( y );
        for( int x = 0; x < wid; x ++ ) {
            if( lValidData[x] == 0 ) {
                // find left first valid pixel
                int lFirst = x;
                int lFind = 0;
                while( lFirst >= 0 ) {
                    if( lValidData[ lFirst ] ) {
                        lFind = 1;
                        break;
                    }
                    lFirst --;
                }
                int rFind = 0;
                // find right first valid pixel
                int rFirst = x;
                while( rFirst < wid ) {
                    if( lValidData[ rFirst ] ) {
                        rFind = 1;
                        break;
                    }
                    rFirst ++;
                }
                // set x's depth to the lowest one
                if( lFind && rFind ) {
                    if( lDisData[ lFirst ] <= lDisData[ rFirst ] ) {
                        lDisData[ x ] = lDisData[ lFirst ];
                    } else {
                        lDisData[ x ] = lDisData[ rFirst ];
                    }
                } else if( lFind ) {
                    lDisData[ x ] = lDisData[ lFirst ];
                } else if ( rFind ) {
                    lDisData[ x ] = lDisData[ rFirst ];
                }
            }
        }
    }
    // fill right dep
    for( int y = 0; y < hei; y ++ ) {
		uchar* rDisData = ( uchar* ) ( rDis.ptr<uchar>( y ) );
		uchar* rValidData = ( uchar* ) rValid.ptr<uchar>( y );
        for( int x = 0; x < wid; x ++ ) {
            if( rValidData[x] == 0 ) {
                // find left first valid pixel
                int lFirst = x;
                int lFind = 0;
                while( lFirst >= 0 ) {
                    if( rValidData[ lFirst ] ) {
                        lFind = 1;
                        break;
                    }
                    lFirst --;
                }
                // find right first valid pixel
                int rFirst = x;
                int rFind = 0;
                while( rFirst < wid ) {
                    if( rValidData[ rFirst ] ) {
                        rFind = 1;
                        break;
                    }
                    rFirst ++;
                }
                if( lFind && rFind ) {
                    // set x's depth to the lowest one
                    if( rDisData[ lFirst ] <= rDisData[ rFirst ] ) {
                        rDisData[ x ] = rDisData[ lFirst ];
                    } else {
                        rDisData[ x ] = rDisData[ rFirst ];
                    }
                } else if( lFind ) {
                    rDisData[ x ] = rDisData[ lFirst ];
                } else if ( rFind )  {
                    rDisData[ x ] = rDisData[ rFirst ];
                }

            }
        }
    }
    return;
}

void wgtMedian(const Mat& lImg, const Mat& rImg, Mat& lDis, Mat& rDis, Mat& lValid, Mat& rValid, const int maxDis)
{
    int hei = lImg.rows;
    int wid = lImg.cols;
    int wndR = MED_SZ / 2;

    double* disHist = new double[maxDis];

    // filter left
    for( int y = 0; y < hei; y ++  ) {
		uchar* lDisData = ( uchar* ) lDis.ptr<uchar>( y );
		float* pL = ( float* ) lImg.ptr<float>( y );
		uchar* lValidData = ( uchar* ) lValid.ptr<uchar>( y );
        for( int x = 0; x < wid; x ++ ) {
            if( lValidData[x] == 0 ) {
                // just filter invalid pixels
                memset( disHist, 0, sizeof( double ) * maxDis );
                double sumWgt = 0.0f;
                // set disparity histogram by bilateral weight
                for( int wy = - wndR; wy <= wndR; wy ++ ) {
                    int qy = ( y + wy + hei ) % hei;
					float* qL = ( float* ) lImg.ptr<float>( qy );
					uchar* qDisData = ( uchar* ) lDis.ptr<uchar>( qy );
                    for( int wx = - wndR; wx <= wndR; wx ++ ) {
                        int qx = ( x + wx + wid ) % wid;
                        int qDep = qDisData[ qx ];
                        if( qDep != 0 ) {
                            double disWgt = wx * wx + wy * wy;
                            double clrWgt =
								( pL[ 3 * x ] - qL[ 3 * qx ] ) * ( pL[ 3 * x ] - qL[ 3 * qx ] ) +
                                ( pL[ 3 * x + 1 ] - qL[ 3 * qx + 1 ] ) * ( pL[ 3 * x + 1 ] - qL[ 3 * qx + 1 ] ) +
                                ( pL[ 3 * x + 2 ] - qL[ 3 * qx + 2 ] ) * ( pL[ 3 * x + 2 ] - qL[ 3 * qx + 2 ] );
                            double biWgt = exp( - disWgt / ( SIG_DIS * SIG_DIS ) - clrWgt / ( SIG_CLR * SIG_CLR ) );
                            disHist[ qDep ] += biWgt;
                            sumWgt += biWgt;
                        }
                    }
                }
                double halfWgt = sumWgt / 2.0f;
                sumWgt = 0.0f;
                int filterDep = 0;
                for( int d = 0; d < maxDis; d ++ ) {
                    sumWgt += disHist[ d ];
                    if( sumWgt >= halfWgt ) {
                        filterDep = d;
                        break;
                    }
                }
                // set new disparity
                lDisData[ x ] = filterDep;
            }
        }
    }

    // filter right depth
    for( int y = 0; y < hei; y ++  ) {
		uchar* rDisData = ( uchar* ) rDis.ptr<uchar>( y );
		float* pR = ( float* ) rImg.ptr<float>( y );
		uchar* rValidData = ( uchar* ) rValid.ptr<uchar>( y );
        for( int x = 0; x < wid; x ++ ) {
            if( rValidData[x] == 0 ) {
                // just filter invalid pixels
                memset( disHist, 0, sizeof( double ) * maxDis );
                double sumWgt = 0.0f;
                // set disparity histogram by bilateral weight
                for( int wy = - wndR; wy <= wndR; wy ++ ) {
                    int qy = ( y + wy + hei ) % hei;
					float* qR = ( float* ) rImg.ptr<float>( qy );
					uchar* qDisData = ( uchar* ) rDis.ptr<uchar>( qy );
                    for( int wx = - wndR; wx <= wndR; wx ++ ) {
                        int qx = ( x + wx + wid ) % wid;
                        int qDep = qDisData[ qx ];
                        if( qDep != 0 ) {
                            double disWgt = wx * wx + wy * wy;
                            disWgt = sqrt( disWgt );
                            double clrWgt =
                                ( pR[ 3 * x ] - qR[ 3 * qx ] ) * ( pR[ 3 * x ] - qR[ 3 * qx ] ) +
                                ( pR[ 3 * x + 1 ] - qR[ 3 * qx + 1 ] ) * ( pR[ 3 * x + 1 ] - qR[ 3 * qx + 1 ] ) +
                                ( pR[ 3 * x + 2 ] - qR[ 3 * qx + 2 ] ) * ( pR[ 3 * x + 2 ] - qR[ 3 * qx + 2 ] );
                            clrWgt = sqrt( clrWgt );
                            double biWgt = exp( - disWgt / ( SIG_DIS * SIG_DIS ) - clrWgt / ( SIG_CLR * SIG_CLR ) );
                            disHist[ qDep ] += biWgt;
                            sumWgt += biWgt;
                        }
                    }
                }
                double halfWgt = sumWgt / 2.0f;
                sumWgt = 0.0f;
                int filterDep = 0;
                for( int d = 0; d < maxDis; d ++ ) {
                    sumWgt += disHist[ d ];
                    if( sumWgt >= halfWgt ) {
                        filterDep = d;
                        break;
                    }
                }
                // set new disparity
                rDisData[ x ] = filterDep;
            }
        }
    }
    return;
}

void *wgtMed_row(void *thread_arg)
{
	//Passing function arguments from wgtMedian_thread
    struct WM_row_TD *t_data;
    t_data = (struct WM_row_TD *) thread_arg;
    //Matricies
    const Mat* Img = t_data->Img;
    Mat* Dis = t_data->Dis;
    //Pointers
    uchar *pValid = t_data->pValid;
    //Variables
    int y = t_data->y;
    int maxDis  = t_data->maxDis;

    int wndR = MED_SZ / 2;
    int hei  = Img->rows;
    int wid = Img->cols;

    double* disHist = new double[maxDis];
	uchar* DisData = (uchar*) Dis->ptr<uchar>(y);
	float* p = (float*) Img->ptr<float>(y);

	for( int x = 0; x < wid; x ++ ) {
		if( pValid[x] == 0 ) {
			// just filter invalid pixels
			memset( disHist, 0, sizeof( double ) * maxDis );
			double sumWgt = 0.0f;
			// set disparity histogram by bilateral weight
			for( int wy = - wndR; wy <= wndR; wy++ ) {
				int qy = ( y + wy + hei ) % hei;
				float* q = ( float* ) Img->ptr<float>( qy );
				uchar* qDisData = ( uchar* ) Dis->ptr<uchar>( qy );
				for( int wx = - wndR; wx <= wndR; wx ++ ) {
					int qx = ( x + wx + wid ) % wid;
					// invalid pixel also used
					int qDep = qDisData[ qx ];
					if( qDep != 0 ) {
						double disWgt = wx * wx + wy * wy;
						double clrWgt =
							( p[ 3 * x ] - q[ 3 * qx ] ) * ( p[ 3 * x ] - q[ 3 * qx ] ) +
							( p[ 3 * x + 1 ] - q[ 3 * qx + 1 ] ) * ( p[ 3 * x + 1 ] - q[ 3 * qx + 1 ] ) +
							( p[ 3 * x + 2 ] - q[ 3 * qx + 2 ] ) * ( p[ 3 * x + 2 ] - q[ 3 * qx + 2 ] );
						double biWgt = exp( - disWgt / ( SIG_DIS * SIG_DIS ) - clrWgt / ( SIG_CLR * SIG_CLR ) );
						disHist[ qDep ] += biWgt;
						sumWgt += biWgt;
					}
				}
			}
			double halfWgt = sumWgt / 2.0f;
			sumWgt = 0.0f;
			int filterDep = 0;
			for( int d = 0; d < maxDis; d ++ ) {
				sumWgt += disHist[ d ];
				if( sumWgt >= halfWgt ) {
					filterDep = d;
					break;
				}
			}
			// set new disparity
			DisData[ x ] = filterDep;
		}
	}
	return (void*)0;
}

void wgtMedian_thread(const Mat& Img, Mat& Dis, Mat& Valid, const int maxDis, const int threads)
{
    int hei = Img.rows;

	//Set up threads for x-loop
    void* status;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t WM_row_threads[hei];
    WM_row_TD WM_row_TD_Array[hei];

	for(int level = 0; level <= hei/threads; level ++)
	{
        //Handle remainder if threads is not power of 2.
	    int block_size = (level < hei/threads) ? threads : (hei%threads);

	    for(int iter=0; iter < block_size; iter++)
	    {
	        int d = level*threads + iter;
			uchar* ValidData = (uchar*) Valid.ptr<uchar>(d);

            WM_row_TD_Array[d] = {&Img, &Dis, ValidData, d, maxDis};
            pthread_create(&WM_row_threads[d], &attr, wgtMed_row, (void *)&WM_row_TD_Array[d]);
            //fprintf(stderr, "WM Filtering Disparity Map @ y = %d\n", d);
	    }
        for(int iter=0; iter < block_size; iter++)
	    {
	        int d = level*threads + iter;
            pthread_join(WM_row_threads[d], &status);
            //fprintf(stderr, "Joining WM Filtering @ y = %d\n", d);
        }
	}
	return;
}

void saveChk(Mat& lValid, Mat& rValid)
{
	int hei = lValid.rows;
	int wid = lValid.cols;

	Mat lChk = Mat::zeros( hei, wid, CV_8UC1 );
	Mat rChk = Mat::zeros( hei, wid, CV_8UC1 );
	for( int y = 0; y < hei; y ++ ) {
		uchar* lChkData = ( uchar* )( lChk.ptr<uchar>( y ) );
		uchar* rChkData = ( uchar* )( rChk.ptr<uchar>( y ) );
		uchar* lValidData = ( uchar* ) lValid.ptr<uchar>( y );
		uchar* rValidData = ( uchar* ) rValid.ptr<uchar>( y );
		for( int x = 0; x < wid; x ++ ) {
			if( lValidData[x] ) {
				lChkData[ x ] = 0;
			} else{
				lChkData[ x ] = 255;
			}

			if( rValidData[x] ) {
				rChkData[ x ] = 0;
			} else{
				rChkData[ x ] = 255;
			}
		}
	}
	imwrite( "l_chk.png", lChk );
	imwrite( "r_chk.png", rChk );

	return;
}

void PP::processDM(const Mat& lImg, const Mat& rImg, Mat& lDisMap, Mat& rDisMap,
					Mat& lValid, Mat& rValid, const int maxDis, int threads)
{
	// color image should be 3x3 median filtered
	// according to weightedMedianMatlab.m from CVPR11

	lrCheck(lDisMap, rDisMap, lValid, rValid);
	//fprintf(stderr, "LR Check Done\n");
	fillInv(lDisMap, rDisMap, lValid, rValid);
	//fprintf(stderr, "Fill Inv Done\n");

	//wgtMedian( lImg, rImg, lDisMap, rDisMap, lValid, rValid, maxDis);
	wgtMedian_thread(lImg, lDisMap, lValid, maxDis, threads);
	wgtMedian_thread(rImg, rDisMap, rValid, maxDis, threads);
	//fprintf(stderr, "Weighted-Median Filter Done\n");

	//lrCheck(lDisMap, rDisMap, lValid, rValid);
	//saveChk(lValid, rValid);
	return;
}
