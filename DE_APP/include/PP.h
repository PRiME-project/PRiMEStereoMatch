/*---------------------------------------------------------------------------
   PP.h - Post Processing Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"

#define MED_SZ 19
#define SIG_CLR 0.1
#define SIG_DIS 9

//
// Weighted-Median Post-processing
//
class PP
{
public:
	PP(void);
	~PP(void);
public:
	void processDM(const Mat& lImg, const Mat& rImg, const int maxDis,
                    Mat& lDisMap, Mat& rDisMap, Mat& lSeg, Mat& lChk);
};

