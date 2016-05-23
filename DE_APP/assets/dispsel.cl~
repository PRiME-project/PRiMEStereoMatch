/*---------------------------------------------------------------------------
   dispsel.cl - OpenCL Disparity Selection Kernel
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/**
 * \brief Disparity Estimation kernel function.
 * \param[in] lcostVol - Calculated pixel cost for Cost Volume.
 * \param[in] rcostVol - Calculated pixel cost for Cost Volume.
 * \param[in] height - Height of the image.
 * \param[in] width - Width of the image.
 * \param[out] ldispMap - Left Disparity Map.
 * \param[out] rdispMap - Right Disparity Map.
 */
__kernel void dispsel(__global const float* restrict lcostVol,
                 	__global const float* restrict rcostVol,
                  	const int height,
                  	const int width,
                  	const int maxDis,
                  	__global char* restrict ldispMap,
                  	__global char* restrict rdispMap)
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
	float minCost = 1e10; //DOUBLE_MAX
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
	*(ldispMap + dispMap_offset + x) = minDis * 4;

	/* *************** Right Disparity Selection ********************** */
	minCost = 1e10;
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
	*(rdispMap + dispMap_offset + x) = minDis * 4;
}
