/*---------------------------------------------------------------------------
   CVF.h - Cost Volume Filter Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"

//
// GIF for Cost Computation
//
class CVF
{
public:

	CVF(void);
	~CVF(void);

	static void *filterCV_thread(void *thread_arg);
	int preprocess(const Mat& Img, Mat* Img_rgb, Mat* mean_Img, Mat* var_Img);
	void filterCV(const Mat* Img_rgb, const Mat* mean_Img, const Mat* var_Img, Mat& costVol);
};
Mat GuidedFilter_cv(const Mat* rgb, const Mat* mean_I, const Mat* var_I, const Mat& p);

//CVF thread data struct
struct filterCV_TD{Mat* Img_rgb; Mat* mean_Img; Mat* var_Img; Mat* costVol;};
