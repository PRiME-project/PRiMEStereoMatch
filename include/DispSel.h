/*---------------------------------------------------------------------------
   DispSel.h - Disparity Selection Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"

class DispSel
{
public:
	DispSel(void);
	~DispSel(void);

	void CVSelect(Mat* costVol, const int maxDis, Mat& dispMap);
	void CVSelect_thread(Mat* costVol, const int maxDis, Mat& dispMap, int threads);
};

struct DS_X_TD{Mat* costVol; Mat* dispMap; int y; int maxDis;};
