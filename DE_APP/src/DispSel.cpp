/*---------------------------------------------------------------------------
   DispSel.cpp - Disparity Selection Code
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "DispSel.h"

DispSel::DispSel(void)
{
    //fprintf(stderr, "Winner-Takes-All Disparity Selection\n" );
}
DispSel::~DispSel(void) {}

void DispSel::CVSelect(Mat* costVol, const int maxDis, Mat& disMap)
{
    int hei = disMap.rows;
    int wid = disMap.cols;

    for(int y = 0; y < hei; y++)
    {
		uchar* disData = ( uchar* ) disMap.ptr<uchar>( y );

        for(int x = 0; x < wid; x++)
        {
            float minCost = DOUBLE_MAX;
            int    minDis  = 0;

            for(int d = 1; d < maxDis; d++)
            {
				float* costData = ( float* )costVol[ d ].ptr<float>( y );
                if( costData[x] < minCost )
                {
                    minCost = costData[x];
                    minDis  = d;
                }
            }
            disData[x] = minDis * 4;
        }
    }
}

