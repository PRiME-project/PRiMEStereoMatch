/*---------------------------------------------------------------------------
   cvf.cl - OpenCL Cost Volume Filtering Kernels
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/

/**
 * \brief Kernel function for element-wise multiplication of 3D and 3D Matricies.
 * \param[in] pIn_a -  First Input Matrix.
 * \param[in] pIn_b -  Second Input Matrix.
 * \param[in] width - Image Width.
 * \param[in] height - Image Height.
 * \param[out] pOut - Output Matrix.
 */

__kernel void EWMul_SameDim(__global const float* pIn_a,
							__global const float* pIn_b,
							const int width,
							const int height,
    	              		__global float* pOut)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int d = get_global_id(2);

    const int offset = (((d * height) + y) * width) + x;

	pOut[offset] = pIn_a[offset] * pIn_b[offset];
}

/**
 * \brief Kernel function for element-wise multiplication of 2D and 3D Matricies.
 * \param[in] pIn_a -  First Input Matrix (2D).
 * \param[in] pIn_b -  Second Input Matrix (3D).
 * \param[in] width - Image Width.
 * \param[in] height - Image Height.
 * \param[out] pOut - Output Matrix.
 */

__kernel void EWMul_DiffDim(__global const float* pIn_a,
							__global const float* pIn_b,
							const int width,
							const int height,
    	              		__global float* pOut)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int d = get_global_id(2);

    const int offset2D = (y * width) + x;
    const int offset3D = (((d * height) + y) * width) + x;

	pOut[offset3D] = pIn_a[offset2D] * pIn_b[offset3D];
}

/**
 * \brief Kernel function for element-wise division of 3D and 3D Matricies.
 * \param[in] pIn_a -  First Input Matrix.
 * \param[in] pIn_b -  Second Input Matrix.
 * \param[in] width - Image Width.
 * \param[in] height - Image Height.
 * \param[out] pOut - Output Matrix.
 */

__kernel void EWDiv_SameDim(__global const float* pIn_a,
							__global const float* pIn_b,
							const int width,
							const int height,
    	              		__global float* pOut)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int d = get_global_id(2);

    const int offset = (((d * height) + y) * width) + x;

	pOut[offset] = pIn_a[offset] / pIn_b[offset];
}

/**
 * \brief Kernel function for channel division of an RGB image
 * \param[in] pImg -  3-channel RGB Input image.
 * \param[in] width - Image Width.
 * \param[in] height - Image Height.
 * \param[out] pIr -  Red channel output.
 * \param[out] pIg -  Green channel output.
 * \param[out] pIb -  Blue channel output.
 */

__kernel void Split(__global const float* pImg,
					const int width,
					__global float* pIr,
					__global float* pIg,
					__global float* pIb)
{
    const int x = get_global_id(0) * 12;
    const int y = get_global_id(1);

    const int offset = (y * width) + x;

	pIr[offset] = pImg[offset];
	pIg[offset] = pImg[offset + 1];
	pIb[offset] = pImg[offset + 2];

	pIr[offset + 1] = pImg[offset + 3];
	pIg[offset + 1] = pImg[offset + 4];
	pIb[offset + 1] = pImg[offset + 5];

	pIr[offset + 2] = pImg[offset + 6];
	pIg[offset + 2] = pImg[offset + 7];
	pIb[offset + 2] = pImg[offset + 8];

	pIr[offset + 3] = pImg[offset + 9];
	pIg[offset + 3] = pImg[offset + 10];
	pIb[offset + 3] = pImg[offset + 11];
}

/**
 * \brief Kernel function for the subtraction of 2 matricies
 * \param[in] pIn_a -  First Input Matrix.
 * \param[in] pIn_b -  Second Input Matrix.
 * \param[in] width - Image Width.
 * \param[in] height - Image Height.
 * \param[out] pOut - Output Matrix.
 */

__kernel void Subtract(__global const float* pIn_a,
						__global const float* pIn_b,
						const int width,
						const int height,
    	              	__global float* pOut)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int d = get_global_id(2);

    const int offset = (((d * height) + y) * width) + x;

	pOut[offset] = pIn_a[offset] - pIn_b[offset];
}

/**
 * \brief Kernel function for the addition of 2 matricies
 * \param[in] pIn_a -  First Input Matrix.
 * \param[in] pIn_b -  Second Input Matrix.
 * \param[in] width - Image Width.
 * \param[in] height - Image Height.
 * \param[out] pOut - Output Matrix.
 */

__kernel void Add(__global const float* pIn_a,
					__global const float* pIn_b,
					const int width,
					const int height,
					__global float* pOut)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int d = get_global_id(2);

    const int offset = (((d * height) + y) * width) + x;

	pOut[offset] = pIn_a[offset] + pIn_b[offset];
}

/**
 * \brief Kernel function for the addition of a constant to a matrix
 * \param[in] pIn_a -  First Input Matrix.
 * \param[in] eps -  Constant offset.
 * \param[in] width - Image Width.
 * \param[in] height - Image Height.
 * \param[out] pOut - Output Matrix.
 */

__kernel void Add_Const(__global const float* pIn_a,
						const float eps,
						const int width,
						const int height,
    	              	__global float* pOut)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int d = get_global_id(2);

    const int offset = (((d * height) + y) * width) + x;

	pOut[offset] = pIn_a[offset] + eps;
}

/**
 * \brief Volume BoxFiltering kernel function.
 * \param[in] pIn -  Input data.
 * \param[in] width - Image Width.
 * \param[in] height - Image Height.
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

