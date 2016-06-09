/*---------------------------------------------------------------------------
   boxfilter_vector16.cl - OpenCL Boxfilter Kernel
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
						const int width,
						const int height,
                  		__global float* pOut)
{
	int load_rows = 9;
	int calc_rows = 1;

    /* [Kernel size] */
    /* Each kernel calculates a single output pixel.
     * column (x) is in the range [0, width].
     * row (y) is in the range [0, height].
     * disparity (d) is in the range [0, maxDis].
     */
    const int x = get_global_id(0) * 16;
    const int y = get_global_id(1) * calc_rows;
    const int d = get_global_id(2);

    const int offset = (((d * height) + y) * width) + x;

	float16 row[9][9];
	float16 res[1];

	for (int r=0; r<load_rows; r++)
	{
			row[r][0] = vload16(0, pIn + offset + width*r);
			row[r][8] = vload16(0, pIn + offset + width*r + 8);
			row[r][1] = (float16)(row[r][0].s12345678, row[r][8].s12345678);
			row[r][2] = (float16)(row[r][0].s23456789, row[r][8].s23456789);
			row[r][3] = (float16)(row[r][0].s3456789a, row[r][8].s3456789a);
			row[r][4] = (float16)(row[r][0].s456789ab, row[r][8].s456789ab);
			row[r][5] = (float16)(row[r][0].s56789abc, row[r][8].s56789abc);
			row[r][6] = (float16)(row[r][0].s6789abcd, row[r][8].s6789abcd);
			row[r][7] = (float16)(row[r][0].s789abcde, row[r][8].s789abcde);
	}

	for (int r=0; r<9; r++)
	{
		for (int c=0; c<9; c++)
		{
			for (int i=0; i<calc_rows; i++)
			{
				res[i] += row[r+i][c];
			}
		}
	}

	for (int i=0; i<calc_rows; i++)
	{
		res[i] /= 81;
		vstore16(res[i], 0, pOut + offset + width*i + 4);
	}
}
