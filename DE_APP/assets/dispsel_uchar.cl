/*---------------------------------------------------------------------------
   dispsel.cl - OpenCL Disparity Selection Kernel
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/**
 * \brief Disparity Estimation kernel function.
 * \param[in] lcostVol - Calculated pixel cost for Cost Volume.
 * \param[in] rcostVol - Calculated pixel cost for Cost Volume.
 * \param[in] height - Height of the image.
 * \param[in] width - Width of the image.
 * \param[out] ldispMap - Left Disparity Map.
 * \param[out] rdispMap - Right Disparity Map.
 */
__kernel void dispsel(__global const char* lcostVol,
                 	__global const char* rcostVol,
                  	const int height,
                  	const int width,
                  	const int maxDis,
                  	__global char* ldispMap,
                  	__global char* rdispMap)
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

	char lminCost = UCHAR_MAX;
	char lminDis = 0;
	char rminCost = UCHAR_MAX;
	char rminDis = 0;

	for(int d = 1; d < maxDis; d++)
	{
    	char lcostData = lcostVol[((d * height) + y) * width + x];
    	char rcostData = rcostVol[((d * height) + y) * width + x];
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
