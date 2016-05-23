/*---------------------------------------------------------------------------
   DispEst.cpp - Disparity Estimation Class
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "DispEst.h"

DispEst::DispEst(Mat l, Mat r, const int d, const int t, bool ocl)
    : lImg(l), rImg(r), maxDis(d), threads(t), useOCL(ocl)
{
    //fprintf(stderr, "Disparity Estimation for Stereo Matching\n" );

    hei = lImg.rows;
    wid = lImg.cols;

    lcostVol = new Mat[maxDis];
    rcostVol = new Mat[maxDis];
    for (int i = 0; i < maxDis; i++)
    {
        lcostVol[i] = Mat::zeros(hei, wid, CV_32FC1);
        rcostVol[i] = Mat::zeros(hei, wid, CV_32FC1);
    }

    lDisMap = Mat::zeros(hei, wid, CV_8UC1);
    rDisMap = Mat::zeros(hei, wid, CV_8UC1);
    lSeg = Mat::zeros(hei, wid, CV_8UC3);
    lChk = Mat::zeros(hei, wid, CV_8UC1);

	//C++ pthreads function constructors
    constructor = new CVC();
    filter = new CVF();
    selector = new DispSel();

    if(useOCL){
		//OpenCL function constructors
		constructor_cl = new CVC_cl(lImg, maxDis); //Parameters required for OpenCL setup
		//constructor_cl_image = new CVC_cli(lImg, maxDis); //Parameters required for OpenCL setup
		filter_cl = new CVF_cl(lImg, maxDis); //Parameters required for OpenCL setup
		selector_cl = new DispSel_cl(lImg, maxDis); //Parameters required for OpenCL setup
    }
}

DispEst::~DispEst(void)
{
    delete [] lcostVol;
    delete [] rcostVol;
    delete constructor;
    delete filter;
    delete selector;
    if(useOCL){
		delete constructor_cl;
		//constructor_cl_image;
		delete filter_cl;
		delete selector_cl;
    }
}

// get left disparity
Mat DispEst::getLDisMap()
{
	return lDisMap;
}
// get right disparity
Mat DispEst::getRDisMap()
{
	return rDisMap;
}

//#############################################################################################################
//# Cost Volume Construction
//#############################################################################################################
void DispEst::CostConst()
{
    //fprintf(stderr,"Cost Construction Underway..\n");
    //CVC* constructor = new CVC();

    // Build Cost Volume
    for( int d = 0; d < maxDis; d ++ )
    {
        constructor->buildCV_left(lImg, rImg, d, lcostVol[d]);
        constructor->buildCV_right(rImg, lImg, d, rcostVol[d]);
    }

    //fprintf(stderr, "Construction Complete\n");
    //delete constructor;
}

void DispEst::CostConst_CPU()
{
    //Set up threads and thread attributes
    void *status;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t BCV_threads[maxDis];
    buildCV_TD buildCV_TD_Array[maxDis];

    //fprintf(stderr,"Cost Construction Underway..\n");
    //CVC* constructor = new CVC();

    for(int level = 0; level <= maxDis/threads; level ++)
	{
	    //Handle remainder if threads is not power of 2.
	    int block_size = (level < maxDis/threads) ? threads : (maxDis%threads);

	    for(int iter=0; iter < block_size; iter++)
	    {
	        int d = level*threads + iter;
            buildCV_TD_Array[d] = {&lImg, &rImg, d, &lcostVol[d]};
            pthread_create(&BCV_threads[d], &attr, CVC::buildCV_left_thread, (void *)&buildCV_TD_Array[d]);
            //printf("Creating BCV L Thread %d\n",d);
	    }
        for(int iter=0; iter < block_size; iter++)
	    {
	        int d = level*threads + iter;
            pthread_join(BCV_threads[d], &status);
            //printf("Joining BCV L Thread %d\n",d);
        }
	}
	for(int level = 0; level <= maxDis/threads; level ++)
	{
        //Handle remainder if threads is not power of 2.
	    int block_size = (level < maxDis/threads) ? threads : (maxDis%threads);

	    for(int iter=0; iter < block_size; iter++)
	    {
	        int d = level*threads + iter;
            buildCV_TD_Array[d] = {&rImg, &lImg, d, &rcostVol[d]};
            pthread_create(&BCV_threads[d], &attr, CVC::buildCV_right_thread, (void *)&buildCV_TD_Array[d]);
	    }
        for(int iter=0; iter < block_size; iter++)
	    {
	        int d = level*threads + iter;
            pthread_join(BCV_threads[d], &status);
        }
	}
    //fprintf(stderr, "Construction Complete\n");
    //delete constructor;
}

void DispEst::CostConst_GPU()
{
    constructor_cl->buildCV(lImg, rImg, lcostVol, rcostVol);
    //constructor_cl_image->buildCV(lImg, rImg, lcostVol, rcostVol);
}

//#############################################################################################################
//# Cost Volume Filtering
//#############################################################################################################
void DispEst::CostFilter()
{
    for(int d = 0; d < maxDis; d++)
	{
        filter->filterCV(lImg, lcostVol[d]);
        filter->filterCV(rImg, rcostVol[d]);
    }
}

void DispEst::CostFilter_CPU()
{
    //fprintf(stderr, "Cost Filtering Underway...\n");
    //Set up threads and thread attributes
    void *status;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_t FCV_threads[maxDis];
    filterCV_TD filterCV_TD_Array[maxDis];

    for(int level = 0; level <= maxDis/threads; level ++)
	{
        //Handle remainder if threads is not power of 2.
	    int block_size = (level < maxDis/threads) ? threads : (maxDis%threads);

	    for(int iter=0; iter < block_size; iter++)
	    {
	        int d = level*threads + iter;
            filterCV_TD_Array[d] = {&lImg, &lcostVol[d]};
            pthread_create(&FCV_threads[d], &attr, CVF::filterCV_thread, (void *)&filterCV_TD_Array[d]);
            //fprintf(stderr, "Filtering Left CV @ Disparity %d\n", d);
	    }
        for(int iter=0; iter < block_size; iter++)
	    {
	        int d = level*threads + iter;
            pthread_join(FCV_threads[d], &status);
            //fprintf(stderr, "Joining Left CV @ Disparity %d\n", d);
        }
	}
	for(int level = 0; level <= maxDis/threads; level ++)
	{
	    //Handle remainder if threads is not power of 2.
	    int block_size = (level < maxDis/threads) ? threads : (maxDis%threads);

	    for(int iter=0; iter < block_size; iter++)
	    {
	        int d = level*threads + iter;
            filterCV_TD_Array[d] = {&rImg, &rcostVol[d]};
            pthread_create(&FCV_threads[d], &attr, CVF::filterCV_thread, (void *)&filterCV_TD_Array[d]);
            //fprintf(stderr, "Filtering Right CV @ Disparity %d\n", d);
	    }
        for(int iter=0; iter < block_size; iter++)
	    {
	        int d = level*threads + iter;
            pthread_join(FCV_threads[d], &status);
            //fprintf(stderr, "Joining Right CV @ Disparity %d\n", d);
        }
	}

    //fprintf(stderr, "Filtering Complete\n");
    //delete filter;
}

void DispEst::CostFilter_GPU() //under construction
{
    //fprintf(stderr, "OpenCL Cost Filtering Underway...\n");
    //CVF_cl* filter_cl = new CVF_cl();
    filter_cl->filterCV(lImg, lcostVol);
    filter_cl->filterCV(rImg, rcostVol);

    //fprintf(stderr, "Filtering Complete\n");
    //delete filter_cl;
}

void DispEst::DispSelect_CPU()
{
    //fprintf(stderr, "Disparity Selection Underway...\n");
    //DispSel* selector = new DispSel();
    //fprintf(stderr, "Left Selection...\n");
    selector->CVSelect(lcostVol, maxDis, lDisMap);
    //fprintf(stderr, "Right Selection...\n");
    selector->CVSelect(rcostVol, maxDis, rDisMap);
    //fprintf(stderr, "Selection Complete\n");
    //delete selector;
}

void DispEst::DispSelect_GPU()
{
    //fprintf(stderr, "Disparity Selection Underway...\n");
    //DispSel_cl* selector_cl = new DispSel_cl();
	//fprintf(stderr, "Selection...\n");
    selector_cl->CVSelect(lcostVol, rcostVol, lDisMap, rDisMap);
    //fprintf(stderr, "Selection Complete\n");
    //delete selector_cl;
}


void DispEst::PostProcess()
{
//    //fprintf(stderr, "Post Processing Underway...\n");
    PP* postProcessor = new PP();
    postProcessor->processDM(lImg, rImg, maxDis, lDisMap, rDisMap, lSeg, lChk);
//    //fprintf(stderr, "Post Processing Complete\n");
}
