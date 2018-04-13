/*---------------------------------------------------------------------------
   DispEst.cpp - Disparity Estimation Class
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
  ---------------------------------------------------------------------------*/
#include "DispEst.h"

DispEst::DispEst(cv::Mat l, cv::Mat r, const int d, int t)
    : lImg(l), rImg(r), maxDis(d), threads(t), subsample_rate(4)
{
#ifdef DEBUG_APP
    std::cout << "Disparity Estimation for Depth Analysis in Stereo Vision Applications." << std::endl;
#endif // DEBUG_APP

    hei = lImg.rows;
    wid = lImg.cols;

    //Global Image Type Checking
	if(lImg.type() == rImg.type())
	{
#ifdef DEBUG_APP
		printf("Data type = %d, CV_32F = %d, CV_8U = %d\n", (lImg.type() & CV_MAT_DEPTH_MASK), CV_32F, CV_8U);
#endif // DEBUG_APP
	} else {
		printf("DE: Error - Left & Right images are of different types.\n");
		exit(1);
	}

    lcostVol = new cv::Mat[maxDis];
    rcostVol = new cv::Mat[maxDis];
    for (int i = 0; i < maxDis; ++i)
    {
		lcostVol[i] = cv::Mat::zeros(hei, wid, CV_32FC1);
		rcostVol[i] = cv::Mat::zeros(hei, wid, CV_32FC1);
    }

	lDisMap = cv::Mat::zeros(hei, wid, CV_8UC1);
	rDisMap = cv::Mat::zeros(hei, wid, CV_8UC1);
	lValid = cv::Mat::zeros(hei, wid, CV_8UC1);
	rValid = cv::Mat::zeros(hei, wid, CV_8UC1);

	printf("Setting up pthreads function constructors\n");
    constructor = new CVC();
    filter = new CVF();
    selector = new DispSel();
    postProcessor = new PP();

	printf("Construction Complete\n");
}

DispEst::~DispEst(void)
{
    delete [] lcostVol;
    delete [] rcostVol;
    delete constructor;
    delete filter;
    delete selector;
    delete postProcessor;
}

int DispEst::setInputImages(cv::Mat leftImg, cv::Mat rightImg)
{
	assert(leftImg.type() == rightImg.type());
	lImg = leftImg;
	rImg = rightImg;
	return 0;
}

int DispEst::setThreads(unsigned int newThreads)
{
	if(newThreads > MAX_CPU_THREADS)
		return -1;

	threads = newThreads;
	return 0;
}

int DispEst::printCV(void)
{
	char filename[20];
	int ret_val = 0;

	for (int i = 0; i < maxDis; ++i)
	{
		if(ret_val = sprintf(filename, "CV/lCV%d.png", i)) return ret_val;
		imwrite(filename, lcostVol[i]*1024*8);
		if(ret_val = sprintf(filename, "CV/rCV%d.png", i)) return ret_val;
		imwrite(filename, rcostVol[i]*1024*8);
	}
	return 0;
}

//#############################################################################################################
//# Cost Volume Construction
//#############################################################################################################
int DispEst::CostConst_OMP()
{
	int ret_val = 0;

	if(ret_val = constructor->preprocess(lImg, lGrdX))
		return ret_val;
	if(ret_val = constructor->preprocess(rImg, rGrdX))
		return ret_val;

	#pragma offload target(mic) in(lImg, rImg)
	{
		// Build Cost Volume
		for(int level = 0; level <= maxDis/threads; ++level)
		{
			//Handle remainder if threads is not power of 2.
			int block_size = (level < maxDis/threads) ? threads : (maxDis%threads);

			#pragma omp parallel for num_threads(threads)
			for(int iter=0; iter < block_size; ++iter)
			{
				int d = level*threads + iter;
				constructor->buildCV_left(lImg, rImg, lGrdX, rGrdX, d, lcostVol[d]);
			}
		}
		for(int level = 0; level <= maxDis/threads; ++level)
		{
			//Handle remainder if threads is not power of 2.
			int block_size = (level < maxDis/threads) ? threads : (maxDis%threads);

			#pragma omp parallel for num_threads(threads)
			for(int iter=0; iter < block_size; ++iter)
			{
				int d = level*threads + iter;
				constructor->buildCV_right(rImg, lImg, rGrdX, lGrdX, d, rcostVol[d]);
			}
		}
	}
    return 0;
}

//#############################################################################################################
//# Cost Volume Filtering
//#############################################################################################################
int DispEst::CostFilter_FGF_OMP()
{
    __attribute__((target(mic))) FastGuidedFilter fgf_left(lImg, GIF_R_WIN, GIF_EPS, subsample_rate);
    __attribute__((target(mic))) FastGuidedFilter fgf_right(rImg, GIF_R_WIN, GIF_EPS, subsample_rate);

	#pragma offload target(mic)
	{
		// Filter Cost Volume
		for(int level = 0; level <= maxDis/threads; ++level)
		{
			//Handle remainder if threads is not power of 2.
			int block_size = (level < maxDis/threads) ? threads : (maxDis%threads);

			#pragma omp parallel for num_threads(threads)
			for(int iter=0; iter < block_size; ++iter)
			{
				int d = level*threads + iter;
				lcostVol[d] = fgf_left.filter(lcostVol[d]);
			}
		}
		for(int level = 0; level <= maxDis/threads; ++level)
		{
			//Handle remainder if threads is not power of 2.
			int block_size = (level < maxDis/threads) ? threads : (maxDis%threads);

			#pragma omp parallel for num_threads(threads)
			for(int iter=0; iter < block_size; ++iter)
			{
				int d = level*threads + iter;
				rcostVol[d] = fgf_right.filter(rcostVol[d]);
			}
		}
	}
	return 0;
}

int DispEst::DispSelect_CPU()
{
    //printf("Left Selection...\n");
    selector->CVSelect(lcostVol, maxDis, threads, lDisMap);
    //selector->CVSelect_thread(lcostVol, maxDis, lDisMap, threads);

    //printf("Right Selection...\n");
    selector->CVSelect(rcostVol, maxDis, threads, rDisMap);
    //selector->CVSelect_thread(rcostVol, maxDis, rDisMap, threads);
	return 0;
}

int DispEst::PostProcess_CPU()
{
    //printf("Post Processing Underway...\n");
    postProcessor->processDM(lImg, rImg, lDisMap, rDisMap, lValid, rValid, maxDis, threads);
    //printf("Post Processing Complete\n");
	return 0;
}
