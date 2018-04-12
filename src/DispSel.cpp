/*---------------------------------------------------------------------------
   DispSel.cpp - Disparity Selection Code
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
  ---------------------------------------------------------------------------*/
#include "DispSel.h"

DispSel::DispSel()
{
#ifdef DEBUG_APP
		std::cout <<  "Winner-Takes-All Disparity Selection." << std::endl;
#endif // DEBUG_APP
}
DispSel::~DispSel() {}

int DispSel::CVSelect(cv::Mat* costVol, const unsigned int maxDis, const int threads, cv::Mat& dispMap)
{
    unsigned int hei = dispMap.rows;
    unsigned int wid = dispMap.cols;

	#pragma omp parallel for num_threads(threads)
    for(unsigned int y = 0; y < hei; ++y)
    {
		for(unsigned int x = 0; x < wid; ++x)
		{
			float minCost = DBL_MAX;
			int minDis = 0;

			for(unsigned int d = 1; d < maxDis; ++d)
			{
				float* costData = (float*)costVol[d].ptr<float>(y);
				if(costData[x] < minCost)
				{
					minCost = costData[x];
					minDis = d;
				}
			}
			dispMap.at<unsigned char>(y,x) = minDis;
		}
    }
    return 0;
}
