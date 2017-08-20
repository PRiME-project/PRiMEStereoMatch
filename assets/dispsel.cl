/*---------------------------------------------------------------------------
   dispsel.cl - OpenCL Disparity Selection Kernel
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/

/**
 * \brief Disparity Selection kernel function.
 * \param[in] lcostVol - Calculated pixel cost for Cost Volume.
 * \param[in] rcostVol - Calculated pixel cost for Cost Volume.
 * \param[in] height - Height of the image.
 * \param[in] width - Width of the image.
 * \param[out] ldispMap - Left Disparity Map.
 * \param[out] rdispMap - Right Disparity Map.
 */
__kernel void dispsel_uchar(__global const uchar* lcostVol,
                 			__global const uchar* rcostVol,
                  			const int height,
                  			const int width,
                  			const int maxDis,
                  			__global uchar* ldispMap,
                  			__global uchar* rdispMap)
{
    /* [Kernel size] */
    /*
     * Each kernel calculates a single output pixels in the same row.
     * column (x) is in the range [0, width].
     * row (y) is in the range [0, height].
     */
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int offset = (y * width) + x;

	uchar lminCost = UCHAR_MAX;
	uchar lminDis = 0;
	uchar rminCost = UCHAR_MAX;
	uchar rminDis = 0;

	for(int d = 1; d < maxDis; d++)
	{
    	uchar lcostData = lcostVol[((d * height) + y) * width + x];
    	uchar rcostData = rcostVol[((d * height) + y) * width + x];
		if(lcostData < lminCost)
		{
			lminCost = lcostData;
			lminDis = d;
		}
		if(rcostData < rminCost)
		{
			rminCost = rcostData;
			rminDis = d;
		}
	}
	ldispMap[offset] = lminDis;
	rdispMap[offset] = rminDis;
}

/**
 * \brief Disparity Estimation kernel function.
 * \param[in] lcostVol - Calculated pixel cost for Cost Volume.
 * \param[in] rcostVol - Calculated pixel cost for Cost Volume.
 * \param[in] height - Height of the image.
 * \param[in] width - Width of the image.
 * \param[out] ldispMap - Left Disparity Map.
 * \param[out] rdispMap - Right Disparity Map.
 */
__kernel void dispsel_float(__global const float* lcostVol,
                 			__global const float* rcostVol,
                  			const int height,
                  			const int width,
                  			const int maxDis,
                  			__global uchar* ldispMap,
                  			__global uchar* rdispMap)
{
    /* [Kernel size] */
    /*
     * Each kernel calculates a single output pixels in the same row.
     * column (x) is in the range [0, width].
     * row (y) is in the range [0, height].
     */
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int dispMap_offset = y * width;
	int costVol_offset;

    /* *************** Left Disparity Selection ********************** */
	float minCost = 1e5f;
	char minDis = 0;

	for(int d = 1; d < maxDis; d++)
	{
    	float costData = *(lcostVol + ((d * height) + y) * width + x);
		if(costData < minCost)
		{
			minCost = costData;
			minDis = d;
		}
	}
	*(ldispMap + dispMap_offset + x) = minDis;

	/* *************** Right Disparity Selection ********************** */
	minCost = 1e5f;
	minDis = 0;

	for(int d = 1; d < maxDis; d++)
	{
    	float costData = *(rcostVol + ((d * height) + y) * width + x);
		if(costData < minCost)
		{
			minCost = costData;
			minDis = d;
		}
	}
	*(rdispMap + dispMap_offset + x) = minDis;
}
