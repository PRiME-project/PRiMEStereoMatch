/*---------------------------------------------------------------------------
   CVF.h - Cost Volume Filter Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"

#define R_WIN 9
#define EPS 0.0001

//
// GIF for Cost Computation
//
class CVF
{
public:

	CVF(void);
	~CVF(void);

	static void *filterCV_thread(void *thread_arg);

	void filterCV(const Mat& Img, Mat& costVol);
};

//CVF thread data struct
struct filterCV_TD{Mat* guideImg; Mat* costVol;};

Mat GuidedFilter_cv(const Mat& I, const Mat& p);
Mat GuidedFilter_cv_v2(const Mat& I, const Mat& p);
