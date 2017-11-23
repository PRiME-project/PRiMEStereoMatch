/*---------------------------------------------------------------------------
   PP.h - Post Processing Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "JointWMF.h"

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

	void processDM(const Mat& lImg, const Mat& rImg, Mat& lDisMap, Mat& rDisMap,
					Mat& lValid, Mat& rValid, const int maxDis, int threads);
};

struct WM_row_TD{const Mat* Img; Mat* Dis; uchar *pValid; int y; int maxDis;};

