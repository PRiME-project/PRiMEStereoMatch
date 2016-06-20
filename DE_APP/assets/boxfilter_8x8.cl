/*---------------------------------------------------------------------------
   cvf.cl - OpenCL Boxfilter Kernel
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
/**
 * \brief Volume BoxFiltering kernel function.
 * \param[in] pIn -  Input data.
 * \param[in] radius - radius of the filter window.
 * \param[out] pOut - Output filtered data.
 */
__kernel void boxfilter(__global const float* pIn,
						const int width,
						const int height,
                  		__global float* pOut)
{
	int load_rows = 17;
	int calc_rows = 8;

    /* [Kernel size] */
    /* Each kernel calculates a single output pixel.
     * column (x) is in the range [0, width].
     * row (y) is in the range [0, height].
     * disparity (d) is in the range [0, maxDis].
     */
    const int x = get_global_id(0) * 8;
    const int y = get_global_id(1) * calc_rows;
    const int d = get_global_id(2);

    const int offset = (((d * height) + y) * width) + x;

	float8 row[17][9];
	float8 res[8];

	#pragma unroll
	for (int r=0; r<load_rows; r++)
	{
			row[r][0] = vload8(0, pIn + offset + width*r);
			row[r][8] = vload8(0, pIn + offset + width*r + 8);
			row[r][1] = (float8)(row[r][0].s1234, row[r][0].s567, row[r][8].s0);
			row[r][2] = (float8)(row[r][0].s2345, row[r][0].s67, row[r][8].s01);
			row[r][3] = (float8)(row[r][0].s3456, row[r][0].s7, row[r][8].s012);
			row[r][4] = (float8)(row[r][0].s4567, row[r][8].s0123);
			row[r][5] = (float8)(row[r][0].s567, row[r][8].s0123, row[r][8].s4);
			row[r][6] = (float8)(row[r][0].s67, row[r][8].s0123, row[r][8].s45);
			row[r][7] = (float8)(row[r][0].s7, row[r][8].s0123, row[r][8].s456);
	}

	#pragma unroll
	for (int i=0; i<calc_rows; i++)
	{
		#pragma unroll
		for (int r=0; r<9; r++)
		{
			#pragma unroll
			for (int c=0; c<9; c++)
			{
				res[i] += row[r+i][c];
			}
		}
		res[i] /= 81;
	}

	#pragma unroll
	for (int i=0; i<calc_rows; i++)
	{
		vstore8(res[i], 0, pOut + offset + width*i + 4);
	}
}

