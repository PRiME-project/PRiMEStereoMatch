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
#define BORDER_CONSTANT 1.0

//#define TAU_1 0.7
//#define TAU_2 0.2
#define TAU_1 0.028
#define TAU_2 0.008
#define ALPHA 0.9

//
// TAD + GRD for Cost Computation
//
class CVC
{
public:
    CVC(void);
    ~CVC(void);

	static void *buildCV_left_thread(void *thread_arg);
	static void *buildCV_right_thread(void *thread_arg);

    void buildCV_left(const Mat& lImg, const Mat& rImg, const int d, Mat& costVol);
    void buildCV_right(const Mat& lImg, const Mat& rImg, const int d, Mat& costVol);
};

//CVC thread data struct
struct buildCV_TD{Mat* lImg; Mat* rImg; int d; Mat* costVol; cpu_set_t en_cpu;};
