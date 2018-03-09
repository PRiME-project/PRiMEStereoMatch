/*---------------------------------------------------------------------------
   DispEst.cpp - Disparity Estimation Class
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
  ---------------------------------------------------------------------------*/
#include "DispEst.h"

DispEst::DispEst(Mat l, Mat r, const int d, const int t, bool ocl)
    : lImg(l), rImg(r), maxDis(d), threads(t), useOCL(ocl)
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

//	lcostVol_cvc = Mat::zeros(hei, wid*maxDis, CV_32FC1);
//	rcostVol_cvc = Mat::zeros(hei, wid*maxDis, CV_32FC1);
    lcostVol = new Mat[maxDis];
    rcostVol = new Mat[maxDis];
    for (int i = 0; i < maxDis; ++i)
    {
		lcostVol[i] = Mat::zeros(hei, wid, CV_32FC1);
		rcostVol[i] = Mat::zeros(hei, wid, CV_32FC1);
    }

//    lImg_rgb = new Mat[3];
//    rImg_rgb = new Mat[3];
//    mean_lImg = new Mat[3];
//    mean_rImg = new Mat[3];
//    var_lImg = new Mat[6];
//    var_rImg = new Mat[6];

	lDisMap = Mat::zeros(hei, wid, CV_8UC1);
	rDisMap = Mat::zeros(hei, wid, CV_8UC1);
	lValid = Mat::zeros(hei, wid, CV_8UC1);
	rValid = Mat::zeros(hei, wid, CV_8UC1);

	printf("Setting up pthreads function constructors\n");
    constructor = new CVC();
    filter = new CVF();
    selector = new DispSel();
    postProcessor = new PP();

    if(useOCL)
    {
		printf("Setting up OpenCL Environment\n");
		//OpenCL Setup
		context = 0;
		commandQueue = 0;
		device = 0;
		numberOfMemoryObjects = 12;
		for(int m = 0; m < (int)numberOfMemoryObjects; m++)
			memoryObjects[m] = 0;

		//cl_int numComputeUnits = 2;
		if (!createContext(&context))
		//if (!createSubDeviceContext(&context, numComputeUnits)) //Device Fission is not supported on Xeon Phi
		{
			cleanUpOpenCL(context, commandQueue, NULL, NULL, memoryObjects, numberOfMemoryObjects);
			cerr << "Failed to create an OpenCL context. " << __FILE__ << ":"<< __LINE__ << endl;
		}
		if (!createCommandQueue(context, &commandQueue, &device))
		{
			cleanUpOpenCL(context, commandQueue, NULL, NULL, memoryObjects, numberOfMemoryObjects);
			cerr << "Failed to create the OpenCL command queue. " << __FILE__ << ":"<< __LINE__ << endl;
		}

		width = (cl_int)wid;
		height = (cl_int)hei;
		channels = (cl_int)lImg.channels();

		//OpenCL Buffers that are type dependent (in accending size order)
//		if(imgType == CV_32F)
//		{
			bufferSize_2D = width * height * sizeof(cl_float);
			bufferSize_3D = width * height * maxDis * sizeof(cl_float);
//		}
//		else if(imgType == CV_8U)
//		{
//			bufferSize_2D = width * height * sizeof(cl_uchar);
//			bufferSize_3D = width * height * maxDis * sizeof(cl_uchar);
//		}
		//OpenCL Buffers that are always required
		bufferSize_2D_8UC1 = width * height * sizeof(cl_uchar);

		/* Create buffers for the left and right images, gradient data, cost volume, and disparity maps. */
		bool createMemoryObjectsSuccess = true;
		memoryObjects[CVC_LIMGR] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_2D, NULL, &errorNumber);
		createMemoryObjectsSuccess &= checkSuccess(errorNumber);
		memoryObjects[CVC_LIMGG] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_2D, NULL, &errorNumber);
		createMemoryObjectsSuccess &= checkSuccess(errorNumber);
		memoryObjects[CVC_LIMGB] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_2D, NULL, &errorNumber);
		createMemoryObjectsSuccess &= checkSuccess(errorNumber);

		memoryObjects[CVC_RIMGR] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_2D, NULL, &errorNumber);
		createMemoryObjectsSuccess &= checkSuccess(errorNumber);
		memoryObjects[CVC_RIMGG] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_2D, NULL, &errorNumber);
		createMemoryObjectsSuccess &= checkSuccess(errorNumber);
		memoryObjects[CVC_RIMGB] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_2D, NULL, &errorNumber);
		createMemoryObjectsSuccess &= checkSuccess(errorNumber);

		memoryObjects[CVC_LGRDX] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_2D, NULL, &errorNumber);
		createMemoryObjectsSuccess &= checkSuccess(errorNumber);
		memoryObjects[CVC_RGRDX] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_2D, NULL, &errorNumber);
		createMemoryObjectsSuccess &= checkSuccess(errorNumber);

		memoryObjects[CV_LCV] = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize_3D, NULL, &errorNumber);
		createMemoryObjectsSuccess &= checkSuccess(errorNumber);
		memoryObjects[CV_RCV] = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize_3D, NULL, &errorNumber);
		createMemoryObjectsSuccess &= checkSuccess(errorNumber);

		memoryObjects[DS_LDM] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_2D_8UC1, NULL, &errorNumber);
		createMemoryObjectsSuccess &= checkSuccess(errorNumber);
		memoryObjects[DS_RDM] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_2D_8UC1, NULL, &errorNumber);
		createMemoryObjectsSuccess &= checkSuccess(errorNumber);
		if (!createMemoryObjectsSuccess)
		{
			cleanUpOpenCL(context, commandQueue, NULL, NULL, memoryObjects, numberOfMemoryObjects);
			std::cerr << "Failed to create OpenCL buffers. " << __FILE__ << ":"<< __LINE__ << std::endl;
		}

		printf("Setting up OpenCL function constructors\n");
		//OpenCL function constructors
		constructor_cl  = new CVC_cl(&context, &commandQueue, device, &lImg, maxDis);
		filter_cl       = new CVF_cl(&context, &commandQueue, device, &lImg, maxDis);
		selector_cl = new DispSel_cl(&context, &commandQueue, device, &lImg, maxDis);
    }

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

    if(useOCL){
		delete constructor_cl;
		delete filter_cl;
		delete selector_cl;

		for(int m = 0; m < (int)numberOfMemoryObjects; ++m)
			clReleaseMemObject(memoryObjects[m]);
    }
}

void DispEst::printCV(void)
{
	char filename[20];

	for (int i = 0; i < maxDis; ++i)
	{
		sprintf(filename, "CV/lCV%d.png", i);
		imwrite(filename, lcostVol[i]*1024*8);
		sprintf(filename, "CV/rCV%d.png", i);
		imwrite(filename, rcostVol[i]*1024*8);
	}
	return;
}

//#############################################################################################################
//# Cost Volume Construction
//#############################################################################################################
void DispEst::CostConst()
{
	float start_time;
	constructor->preprocess(lImg, lGrdX);
	constructor->preprocess(rImg, rGrdX);

	start_time = get_rt();
    // Build Cost Volume
	#pragma omp parallel for
    for( int d = 0; d < maxDis; ++d)
    {
        constructor->buildCV_left(lImg, rImg, lGrdX, rGrdX, d, lcostVol[d]);
    }
	#pragma omp parallel for
    for( int d = 0; d < maxDis; ++d)
    {
        constructor->buildCV_right(rImg, lImg, rGrdX, lGrdX, d, rcostVol[d]);
    }
	std::cout << "Time (ms) = " << (get_rt() - start_time)/1000 << std::endl;
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

	constructor->preprocess(lImg, lGrdX);
	constructor->preprocess(rImg, rGrdX);

    for(int level = 0; level <= maxDis/threads; ++level)
	{
	    //Handle remainder if threads is not power of 2.
	    int block_size = (level < maxDis/threads) ? threads : (maxDis%threads);

	    for(int iter=0; iter < block_size; ++iter)
	    {
	        int d = level*threads + iter;
            buildCV_TD_Array[d] = {&lImg, &rImg, &lGrdX, &rGrdX, d, &lcostVol[d]};
            pthread_create(&BCV_threads[d], &attr, CVC::buildCV_left_thread, (void *)&buildCV_TD_Array[d]);
	    }
        for(int iter=0; iter < block_size; ++iter)
	    {
	        int d = level*threads + iter;
            pthread_join(BCV_threads[d], &status);
        }
	}
	for(int level = 0; level <= maxDis/threads; ++level)
	{
        //Handle remainder if threads is not power of 2.
	    int block_size = (level < maxDis/threads) ? threads : (maxDis%threads);

	    for(int iter=0; iter < block_size; ++iter)
	    {
	        int d = level*threads + iter;
            buildCV_TD_Array[d] = {&rImg, &lImg, &rGrdX, &lGrdX, d, &rcostVol[d]};
            pthread_create(&BCV_threads[d], &attr, CVC::buildCV_right_thread, (void *)&buildCV_TD_Array[d]);
	    }
        for(int iter=0; iter < block_size; ++iter)
	    {
	        int d = level*threads + iter;
            pthread_join(BCV_threads[d], &status);
        }
	}
}

void DispEst::CostConst_GPU()
{
	constructor_cl->buildCV(lImg, rImg, memoryObjects);
}

//#############################################################################################################
//# Cost Volume Filtering
//#############################################################################################################
void DispEst::CostFilter_FGF()
{
    FastGuidedFilter *fgf_left = new FastGuidedFilter(lImg, GIF_R_WIN, GIF_EPS_32F, SUBSAMPLE_RATE);
    FastGuidedFilter *fgf_right = new FastGuidedFilter(rImg, GIF_R_WIN, GIF_EPS_32F, SUBSAMPLE_RATE);

	#pragma omp parallel for
    for(int d = 0; d < maxDis; ++d){
        fgf_left->filter(lcostVol[d]);
    }

	#pragma omp parallel for
    for(int d = 0; d < maxDis; ++d){
        fgf_right->filter(rcostVol[d]);
    }
}

//TODO: Port FGF code to GPU
void DispEst::CostFilter_GPU()
{
    //printf("OpenCL Cost Filtering Underway...\n");
    filter_cl->preprocess(&memoryObjects[CVC_LIMGR], &memoryObjects[CVC_LIMGG], &memoryObjects[CVC_LIMGB]);
    filter_cl->filterCV(&memoryObjects[CV_LCV]);
    filter_cl->preprocess(&memoryObjects[CVC_RIMGR], &memoryObjects[CVC_RIMGG], &memoryObjects[CVC_RIMGB]);
    filter_cl->filterCV(&memoryObjects[CV_RCV]);
    //printf("Filtering Complete\n");
}


void DispEst::DispSelect_CPU()
{
    //printf("Left Selection...\n");
    selector->CVSelect(lcostVol, maxDis, lDisMap);
    //selector->CVSelect_thread(lcostVol, maxDis, lDisMap, threads);

    //printf("Right Selection...\n");
    selector->CVSelect(rcostVol, maxDis, rDisMap);
    //selector->CVSelect_thread(rcostVol, maxDis, rDisMap, threads);
}

void DispEst::DispSelect_GPU()
{
	//printf("Left & Right Selection...\n");
	selector_cl->CVSelect(memoryObjects, lDisMap, rDisMap);
}

void DispEst::PostProcess_CPU()
{
    //printf("Post Processing Underway...\n");
    postProcessor->processDM(lImg, rImg, lDisMap, rDisMap, lValid, rValid, maxDis, threads);
    //printf("Post Processing Complete\n");
}

void DispEst::PostProcess_GPU()
{
    //printf("Post Processing Underway...\n");
    postProcessor->processDM(lImg, rImg, lDisMap, rDisMap, lValid, rValid, maxDis, threads);
    //printf("Post Processing Complete\n");
}
