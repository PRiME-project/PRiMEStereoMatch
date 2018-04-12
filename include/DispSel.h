/*---------------------------------------------------------------------------
   DispSel.h - Disparity Selection Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"

class DispSel
{
public:
	DispSel();
	~DispSel();

	int CVSelect(cv::Mat* costVol, const unsigned int maxDis, const int threads, cv::Mat& dispMap);
};

