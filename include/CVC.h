/*---------------------------------------------------------------------------
   CVC.h - Cost Volume Construction Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"

// CVPR 11
#define BORDER_THRES 0.011764
#define BORDER_CONSTANT_32F 1.0
#define BORDER_CONSTANT_8U  UCHAR_MAX

//#define TAU_1 0.7
//#define TAU_2 0.2
#define TAU_1_32F 0.028
#define TAU_2_32F 0.008
#define TAU_1_16U 1825
#define TAU_2_16U 524

#define ALPHA_32F 0.9f
#define ALPHA_16U 0.9

//
// TAD + GRD for Cost Computation
//
class CVC
{
public:
    CVC(void);
    ~CVC(void);

	void preprocess(const Mat& Img, Mat& GrdX);

	static void *buildCV_left_thread(void *thread_arg);
	static void *buildCV_right_thread(void *thread_arg);

	void buildCV_left(const Mat& lImg, const Mat& rImg, const Mat& lGrdX, const Mat& rGrdX, const int d, Mat& costVol);
	void buildCV_right(const Mat& lImg, const Mat& rImg, const Mat& lGrdX, const Mat& rGrdX, const int d, Mat& costVol);
};

//CVC thread data struct
struct buildCV_TD{Mat* lImg; Mat* rImg; Mat* lGrdX; Mat* rGrdX; int d; Mat* costVol;};
