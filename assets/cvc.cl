/*---------------------------------------------------------------------------
   cvc.cl - OpenCL Cost Volume Construction Kernel
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define UC16M (uchar16)UCHAR_MAX
#define US16M (ushort16)USHRT_MAX

#define TAU_1_32F 0.028f
#define TAU_2_32F 0.008f

//cvc_uchar_v16:
#define ALPHA_US ((ushort16)9/(ushort16)10)
#define TAU_1_US16 (ushort16)1835 //0.028 * USHRT_MAX
#define TAU_2_US16 (ushort16)524 //0.008 * USHRT_MAX

//cvc_uchar_nv:
#define ALPHA 0.9f
#define TAU_1_US 1835 //0.028 * USHRT_MAX
#define TAU_2_US 524 //0.008 * USHRT_MAX

/**
 * \brief Cost Volume Construction kernel function.
 * \param[in] lImg - Left Input image data.
 * \param[in] rImg - Right Input image data.
 * \param[in] lGrdX - Left Input X dim gradient data.
 * \param[in] rGrdX - Right Input X dim gradient data.
 * \param[in] height - Height of the image.
 * \param[in] width - Width of the image.
 * \param[out] lcostVol - Calculated pixel cost for Cost Volume.
 * \param[out] rcostVol - Calculated pixel cost for Cost Volume.
 */

__kernel void cvc_uchar_vx(__global const uchar* lImgR,
							__global const uchar* lImgG,
							__global const uchar* lImgB,
					        __global const uchar* rImgR,
					        __global const uchar* rImgG,
					        __global const uchar* rImgB,
					        __global const uchar* lGrdX,
							__global const uchar* rGrdX,
						      const int height,
						      const int width,
						      __global uchar* lcostVol,
						      __global uchar* rcostVol)
{
    /* [Kernel size] */
    /* Each kernel calculates a single output pixels in the same row.
     * column (x) is in the range [0, width].
     * row (y) is in the range [0, height]. */

//    const int x = get_global_id(0);
    const int y = get_global_id(0);
    const int d = get_global_id(2);

    /* Offset calculates the position in the linear data for the row and the column. */
    const int offset = y * width;
    const int costVol_offset = ((d * height) + y) * width;

	/* *************** Left to Right Cost Volume Construction ********************** */
    ushort clrDiff, grdDiff;

    for(int x = 0; x < d; x++)
    {
        // three color diff at img boundary
        clrDiff = (abs(lImgR[offset + x] - UCHAR_MAX) 
				+ abs(lImgG[offset + x] - UCHAR_MAX) 
				+ abs(lImgB[offset + x] - UCHAR_MAX))/3;
        // gradient diff
        grdDiff = abs(lGrdX[offset + x] - UCHAR_MAX);
		
		clrDiff = clrDiff > TAU_1_US ? TAU_1_US : clrDiff; 
		grdDiff = grdDiff > TAU_2_US ? TAU_2_US : grdDiff; 
		lcostVol[costVol_offset + x] = (uchar)( ALPHA * clrDiff + (1-ALPHA) * grdDiff );
	}
    for(int x = d; x < width; x++)
    {
        // three color diff
        clrDiff = (abs(lImgR[offset + x] - rImgR[offset + x - d]) 
				+ abs(lImgG[offset + x] - rImgG[offset + x - d]) 
				+ abs(lImgB[offset + x] - rImgB[offset + x - d]))/3;
        // gradient diff
        grdDiff = abs(lGrdX[offset + x] - rGrdX[offset + x - d]);

		clrDiff = clrDiff > TAU_1_US ? TAU_1_US : clrDiff; 
		grdDiff = grdDiff > TAU_2_US ? TAU_2_US : grdDiff; 
		lcostVol[costVol_offset + x] = (uchar)( ALPHA * clrDiff + (1-ALPHA) * grdDiff );
    }

	/* *************** Right to Left Cost Volume Construction ********************** */

    for(int x = 0; x < width - d; x++)
    {
        // three color diff
        clrDiff = (abs(rImgR[offset + x] - lImgR[offset + x + d]) 
				+ abs(rImgG[offset + x] - lImgG[offset + x  + d]) 
				+ abs(rImgB[offset + x] - lImgB[offset + x  + d])) * 0.333f;
        // gradient diff
        grdDiff = abs(rGrdX[offset] - lGrdX[offset + x  + d]);

		clrDiff = clrDiff > TAU_1_US ? TAU_1_US : clrDiff; 
		grdDiff = grdDiff > TAU_2_US ? TAU_2_US : grdDiff; 
		rcostVol[costVol_offset + x] = (uchar)( ALPHA * clrDiff + (1-ALPHA) * grdDiff );
    }
    for(int x = width - d; x < width; x++)
	{
        // three color diff at img boundary
        clrDiff = (abs(rImgR[offset + x] - UCHAR_MAX) 
				+ abs(rImgG[offset + x] - UCHAR_MAX) 
				+ abs(rImgB[offset + x] - UCHAR_MAX)) * 0.333f;
        // gradient diff
        grdDiff = abs(rGrdX[offset + x] - UCHAR_MAX);

		clrDiff = clrDiff > TAU_1_US ? TAU_1_US : clrDiff; 
		grdDiff = grdDiff > TAU_2_US ? TAU_2_US : grdDiff; 
		rcostVol[costVol_offset + x] = (uchar)( ALPHA * clrDiff + (1-ALPHA) * grdDiff );
	}
}

/**
 * \brief Cost Volume Construction kernel function.
 * \param[in] lImg - Left Input image data.
 * \param[in] rImg - Right Input image data.
 * \param[in] lGrdX - Left Input X dim gradient data.
 * \param[in] rGrdX - Right Input X dim gradient data.
 * \param[in] height - Height of the image.
 * \param[in] width - Width of the image.
 * \param[out] lcostVol - Calculated pixel cost for Cost Volume.
 * \param[out] rcostVol - Calculated pixel cost for Cost Volume.
 */

__kernel void cvc_uchar_v16(__global const uchar* lImgR,
							__global const uchar* lImgG,
							__global const uchar* lImgB,
					        __global const uchar* rImgR,
					        __global const uchar* rImgG,
					        __global const uchar* rImgB,
					        __global const uchar* lGrdX,
							__global const uchar* rGrdX,
				          	const int height,
				          	const int width,
				          	__global uchar* lcostVol,
				          	__global uchar* rcostVol)
{
    /* [Kernel size] */
    /* Each kernel calculates a single output pixels in the same row.
     * column (x) is in the range [0, width].
     * row (y) is in the range [0, height].
     */
    const int x = get_global_id(0) * 16;
    const int y = get_global_id(1);
    const int d = get_global_id(2);

    /* Offset calculates the position in the linear data for the row and the column. */
    const int offset = (y * width) + x;
    const int costVol_offset = ((d * height) + y) * width + x;

	/* *************** Left to Right Cost Volume Construction ********************** */
    // Set addresses for lImg, rImg, lGrdX and rGrdX.
    uchar16 lCR = vload16(0, lImgR + offset);
    uchar16 lCG = vload16(0, lImgG + offset);
    uchar16 lCB = vload16(0, lImgB + offset);
    uchar16 lGX = vload16(0, lGrdX + offset);
	uchar16 rCR, rCG, rCB, rGX;

    ushort16 clrDiff, grdDiff;
	
    if(x - d >= 0)
    {
    	rCR = vload16(0, rImgR + offset - d);
    	rCG = vload16(0, rImgG + offset - d);
        rCB = vload16(0, rImgB + offset - d);
    	rGX = vload16(0, rGrdX + offset - d);

        // three color diff
        clrDiff = convert_ushort16(abs(lCR - rCR) + abs(lCG - rCG) + abs(lCB - rCB))/(ushort)3;
        // gradient diff
        grdDiff = convert_ushort16(abs(lGX - rGX));
    }
    else
    {
        // three color diff at img boundary
        clrDiff = convert_ushort16(abs(lCR - UC16M) + abs(lCG - UC16M) + abs(lCB - UC16M))/(ushort)3;
        // gradient diff
        grdDiff = convert_ushort16(abs(lGX - UC16M));
    }

    clrDiff = clrDiff > TAU_1_US16 ? TAU_1_US16 : clrDiff; 
    grdDiff = grdDiff > TAU_2_US16 ? TAU_2_US16 : grdDiff; 
	ushort16 cost = clrDiff/(ushort)9*(ushort)10 + grdDiff/(ushort)10;
    vstore16(convert_uchar16(cost), 0, lcostVol + costVol_offset);


	/* *************** Right to Left Cost Volume Construction ********************** */
	// Set addresses for lImg, rImg, lGrdX and rGrdX.
    lCR = vload16(0, rImgR + offset);
    lCG = vload16(0, rImgG + offset);
    lCB = vload16(0, rImgB + offset);
    lGX = vload16(0, rGrdX + offset);

    clrDiff = 0;
    grdDiff = 0;

    if(x + d < width)
    {
    	rCR = vload16(0, lImgR + offset + d);
    	rCG = vload16(0, lImgG + offset + d);
        rCB = vload16(0, lImgB + offset + d);
    	rGX = vload16(0, lGrdX + offset + d);

        // three color diff
        clrDiff = convert_ushort16(abs(lCR - rCR) + abs(lCG - rCG) + abs(lCB - rCB))/(ushort)3;
        // gradient diff
        grdDiff = convert_ushort16(abs(lGX - rGX));
    }
    else
    {
        // three color diff at img boundary
        clrDiff = convert_ushort16(abs(lCR - UC16M) + abs(lCG - UC16M) + abs(lCB - UC16M))/(ushort)3;
        // gradient diff
        grdDiff = convert_ushort16(abs(lGX - UC16M));
    }

    clrDiff = clrDiff > TAU_1_US16 ? TAU_1_US16 : clrDiff; 
    grdDiff = grdDiff > TAU_2_US16 ? TAU_2_US16 : grdDiff; 
	cost = clrDiff/(ushort)9*(ushort)10 + grdDiff/(ushort)10;
    vstore16(convert_uchar16(cost), 0, rcostVol + costVol_offset);
}

/**
 * \brief Cost Volume Construction kernel function.
 * \param[in] lImg - Left Input image data.
 * \param[in] rImg - Right Input image data.
 * \param[in] lGrdX - Left Input X dim gradient data.
 * \param[in] rGrdX - Right Input X dim gradient data.
 * \param[in] height - Height of the image.
 * \param[in] width - Width of the image.
 * \param[out] lcostVol - Calculated pixel cost for Cost Volume.
 * \param[out] rcostVol - Calculated pixel cost for Cost Volume.
 */

__kernel void cvc_uchar_nv(__global const uchar* lImgR,
							__global const uchar* lImgG,
							__global const uchar* lImgB,
					        __global const uchar* rImgR,
					        __global const uchar* rImgG,
					        __global const uchar* rImgB,
					        __global const uchar* lGrdX,
							__global const uchar* rGrdX,
						      const int height,
						      const int width,
						      __global uchar* lcostVol,
						      __global uchar* rcostVol)
{
    /* [Kernel size] */
    /* Each kernel calculates a single output pixels in the same row.
     * column (x) is in the range [0, width].
     * row (y) is in the range [0, height]. */

    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int d = get_global_id(2);

    /* Offset calculates the position in the linear data for the row and the column. */
    const int offset = y * width + x;
    const int costVol_offset = ((d * height) + y) * width + x;

	/* *************** Left to Right Cost Volume Construction ********************** */
    ushort clrDiff, grdDiff;

    if(x >= d)
    {
        // three color diff
        clrDiff = (abs(lImgR[offset] - rImgR[offset - d]) 
				+ abs(lImgG[offset] - rImgG[offset - d]) 
				+ abs(lImgB[offset] - rImgB[offset - d]))/3;
        // gradient diff
        grdDiff = abs(lGrdX[offset] - rGrdX[offset - d]);
    }
    else
    {
        // three color diff at img boundary
        clrDiff = (abs(lImgR[offset] - UCHAR_MAX) 
				+ abs(lImgG[offset] - UCHAR_MAX) 
				+ abs(lImgB[offset] - UCHAR_MAX))/3;
        // gradient diff
        grdDiff = abs(lGrdX[offset] - UCHAR_MAX);
    }

    clrDiff = clrDiff > TAU_1_US ? TAU_1_US : clrDiff; 
    grdDiff = grdDiff > TAU_2_US ? TAU_2_US : grdDiff; 
    lcostVol[costVol_offset] = (uchar)( ALPHA * clrDiff + (1-ALPHA) * grdDiff );


	/* *************** Right to Left Cost Volume Construction ********************** */
    clrDiff = 0;
    grdDiff = 0;

    if(x >= d)
    {
        // three color diff
        clrDiff = (abs(rImgR[offset] - lImgR[offset + d]) 
				+ abs(rImgG[offset] - lImgG[offset + d]) 
				+ abs(rImgB[offset] - lImgB[offset + d]))/3;
        // gradient diff
        grdDiff = abs(rGrdX[offset] - lGrdX[offset + d]);
    }
    else
    {
        // three color diff at img boundary
        clrDiff = (abs(rImgR[offset] - UCHAR_MAX) 
				+ abs(rImgG[offset] - UCHAR_MAX) 
				+ abs(rImgB[offset] - UCHAR_MAX))/3;
        // gradient diff
        grdDiff = abs(rGrdX[offset] - UCHAR_MAX);
    }

    clrDiff = clrDiff > TAU_1_US ? TAU_1_US : clrDiff; 
    grdDiff = grdDiff > TAU_2_US ? TAU_2_US : grdDiff; 
    rcostVol[costVol_offset] = (uchar)( ALPHA * clrDiff + (1-ALPHA) * grdDiff );
}

/**
 * \brief Cost Volume Construction kernel function.
 * \param[in] lImg - Left Input image data.
 * \param[in] rImg - Right Input image data.
 * \param[in] lGrdX - Left Input X dim gradient data.
 * \param[in] rGrdX - Right Input X dim gradient data.
 * \param[in] height - Height of the image.
 * \param[in] width - Width of the image.
 * \param[out] lcostVol - Calculated pixel cost for Cost Volume.
 * \param[out] rcostVol - Calculated pixel cost for Cost Volume.
 */
__kernel void cvc_float_nv(__global const float* lImgR,
							__global const float* lImgG,
							__global const float* lImgB,
					        __global const float* rImgR,
					        __global const float* rImgG,
					        __global const float* rImgB,
					        __global const float* lGrdX,
							__global const float* rGrdX,
                  			const int height,
                  			const int width,
                  			__global float* lcostVol,
                  			__global float* rcostVol)
{
    /* [Kernel size] */
    /*
     * Each kernel calculates a single output pixels in the same row.
     * column (x) is in the range [0, width].
     * row (y) is in the range [0, height].
     */
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int d = get_global_id(2);

    /* Offset calculates the position in the linear data for the row and the column. */
    const int offset = y * width + x;
    const int costVol_offset = ((d * height) + y) * width + x;

	/* *************** Left to Right Cost Volume Construction ********************** */
    float clrDiff, grdDiff;

    if(x >= d)
    {
        // three color diff
        clrDiff = (fabs(lImgR[offset] - rImgR[offset - d]) 
				+ fabs(lImgG[offset] - rImgG[offset - d]) 
				+ fabs(lImgB[offset] - rImgB[offset - d]))/3;
        // gradient diff
        grdDiff = fabs(lGrdX[offset] - rGrdX[offset - d]);
    }
    else
    {
        // three color diff at img boundary
        clrDiff = (fabs(lImgR[offset] - 1.0f) 
				+ fabs(lImgG[offset] - 1.0f) 
				+ fabs(lImgB[offset] - 1.0f))/3;
        // gradient diff
        grdDiff = fabs(lGrdX[offset] - 1.0f);
    }

    clrDiff = clrDiff > TAU_1_32F ? TAU_1_32F : clrDiff; 
    grdDiff = grdDiff > TAU_2_32F ? TAU_2_32F : grdDiff; 
    lcostVol[costVol_offset] = ( ALPHA * clrDiff + (1-ALPHA) * grdDiff );


	/* *************** Right to Left Cost Volume Construction ********************** */
    clrDiff = 0;
    grdDiff = 0;

    if(x >= d)
    {
        // three color diff
        clrDiff = (fabs(rImgR[offset] - lImgR[offset + d]) 
				+ fabs(rImgG[offset] - lImgG[offset + d]) 
				+ fabs(rImgB[offset] - lImgB[offset + d]))/3;
        // gradient diff
        grdDiff = fabs(rGrdX[offset] - lGrdX[offset + d]);
    }
    else
    {
        // three color diff at img boundary
        clrDiff = (fabs(rImgR[offset] - 1.0f) 
				+ fabs(rImgG[offset] - 1.0f) 
				+ fabs(rImgB[offset] - 1.0f))/3;
        // gradient diff
        grdDiff = fabs(rGrdX[offset] - 1.0f);
    }

    clrDiff = clrDiff > TAU_1_32F ? TAU_1_32F : clrDiff; 
    grdDiff = grdDiff > TAU_2_32F ? TAU_2_32F : grdDiff; 
    rcostVol[costVol_offset] = ( ALPHA * clrDiff + (1-ALPHA) * grdDiff );
}

/**
 * \brief Cost Volume Construction kernel function.
 * \param[in] lImg - Left Input image data.
 * \param[in] rImg - Right Input image data.
 * \param[in] lGrdX - Left Input X dim gradient data.
 * \param[in] rGrdX - Right Input X dim gradient data.
 * \param[in] height - Height of the image.
 * \param[in] width - Width of the image.
 * \param[out] lcostVol - Calculated pixel cost for Cost Volume.
 * \param[out] rcostVol - Calculated pixel cost for Cost Volume.
 */
__kernel void cvc_float_v4(__global const float* restrict lImg,
						      __global const float* restrict rImg,
						      __global const float* restrict lGrdX,
						      __global const float* restrict rGrdX,
						      const int height,
						      const int width,
						      __global float* restrict lcostVol,
						      __global float* restrict rcostVol)
{
    /* [Kernel size] */
    /*
     * Each kernel calculates a single output pixels in the same row.
     * column (x) is in the range [0, width].
     * row (y) is in the range [0, height].
     */
    const int x = get_global_id(0) * 4;
    const int y = get_global_id(1);
    const int d = get_global_id(2);

    /* Offset calculates the position in the linear data for the row and the column. */
    int img_offset = y * width + x;
    const int costVol_offset = ((d * height) + y) * width + x;

///	/* *************** Left to Right Cost Volume Construction ********************** */
    // Set addresses for lImg, rImg, lGrdX and rGrdX.
    float16 rC, lC = vload16(0, lImg + (img_offset * 3));
    float4 rGX, lGX = vload4(0, lGrdX + img_offset);

    float4 clrDiff = 0;
    float4 grdDiff = 0;

    if(x - d >= 0)
    {
		img_offset = y * width + x - d;
    	rC = vload16(0, rImg + (img_offset * 3));
    	rGX = vload4(0, rGrdX + img_offset);

        // three color diff
        lC = fabs(lC - rC);
        // gradient diff
        grdDiff = fabs(lGX - rGX);
    }
    else
    {
        // three color diff at img boundary
        lC = fabs(lC - 1.0f); //BORDER_CONSTANT = 1.0
        // gradient diff
        grdDiff = fabs(lGX - 1.0f); //BORDER_CONSTANT = 1.0
    }
    clrDiff = (float4)(lC.s0 + lC.s1 + lC.s2,
					lC.s3 + lC.s4 + lC.s5,
					lC.s6 + lC.s7 + lC.s8,
					lC.s9 + lC.sa + lC.sb) * 0.3333333333f;

    clrDiff = clrDiff > 0.028f ? 0.028f : clrDiff;
    grdDiff = grdDiff > 0.008f ? 0.008f : grdDiff;
    vstore4(0.9f * clrDiff + 0.1f * grdDiff, 0, lcostVol + costVol_offset); //data, offset, addr
//	*(lcostVol + costVol_offset + 4) = 0.9 * clrDiff.s4 + 0.1 * grdDiff.s4;


///	/* *************** Right to Left Cost Volume Construction ********************** */
	// Set addresses for lImg, rImg, lGrdX and rGrdX.
    img_offset = y * width + x;
    lC  = vload16(0, rImg + (img_offset * 3));
    lGX = vload4(0, rGrdX + img_offset);

    grdDiff = 0;

    if(x + d < width)
    {
		img_offset = y * width + x + d;
    	rC  = vload16(0, lImg + (img_offset * 3));
    	rGX = vload4(0, lGrdX + img_offset);

        // three color diff
        lC = fabs(lC - rC);
        // gradient diff
        grdDiff = fabs(lGX - rGX);
    }
    else
    {
        // three color diff at img boundary
        lC = fabs(lC - 1.0f); //BORDER_CONSTANT = 1.0
        // gradient diff
        grdDiff = fabs(lGX - 1.0f); //BORDER_CONSTANT = 1.0
    }
    clrDiff = (float4)(lC.s0 + lC.s1 + lC.s2,
					lC.s3 + lC.s4 + lC.s5,
					lC.s6 + lC.s7 + lC.s8,
					lC.s9 + lC.sa + lC.sb) * 0.3333333333f;

    clrDiff = clrDiff > 0.028f ? 0.028f : clrDiff;
    grdDiff = grdDiff > 0.008f ? 0.008f : grdDiff;
	vstore4(0.9f * clrDiff + 0.1f * grdDiff, 0, rcostVol + costVol_offset); //data, offset, addr
	//*(rcostVol + costVol_offset + 4) = 0.9 * clrDiff.s4 + 0.1 * grdDiff.s4;
}

