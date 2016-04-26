/*---------------------------------------------------------------------------
   boxfilter_novector.cl - OpenCL Boxfilter Kernel
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/

/**
 * \brief Volume BoxFiltering kernel function.
 * \param[in] pIn -  Input data.
 * \param[in] radius - radius of the filter window.
 * \param[out] pOut - Output filtered data.
 */
__kernel void boxfilter(__global const float* pIn,
						const int radius,
                  		__global float* pOut)
{
    /* [Kernel size] */
    /*
     * Each kernel calculates a single output pixel.
     * column (x) is in the range [0, width].
     * row (y) is in the range [0, height].
     * disparity (d) is in the range [0, maxDis].
     */
     /* [Kernel size] */
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int d = get_global_id(2);
    const int width = get_global_size(0);
    const int height = get_global_size(1);

    const int offset = (((d * height) + y) * width) + x;

	float16 row_above = vload16(0, pIn + offset);
	float16 row_middl = vload16(0, pIn + offset + width);
	float16 row_below = vload16(0, pIn + offset + width*2);

//	mem obj + img col & row + centre of filter kernel = x
	*(pOut + offset + width + (radius-1)/2 + 1) = 
		(row_above.s0 + row_above.s1 + row_above.s2 + row_above.s3 + row_above.s4 + row_above.s5 + row_above.s6 + row_above.s7 + row_above.s8 + 
		row_middl.s0 + row_middl.s1 + row_middl.s2 + row_middl.s3 + row_middl.s4 + row_middl.s5 + row_middl.s6 + row_middl.s7 + row_middl.s8 + 
		row_below.s0 + row_below.s1 + row_below.s2 + row_below.s3 + row_below.s4 + row_below.s5 + row_below.s6 + row_below.s7 + row_below.s8)/ 
		(radius*radius);
}
