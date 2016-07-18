/*---------------------------------------------------------------------------
   cvc.cl - OpenCL Cost Volume Construction Kernel
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

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

#define ALPHA 0.9f
#define TAU_1 (uchar)7 //0.028 * USHRT_MAX
#define TAU_2 (uchar)2 //0.008 * USHRT_MAX

__kernel void cvc(__global const uchar* lImgR,
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
    uchar16 lCR = vload16(offset, lImgR);
    uchar16 lCG = vload16(offset, lImgG);
    uchar16 lCB = vload16(offset, lImgB);
    uchar16 lGX = vload16(offset, lGrdX);
	uchar16 rCR, rCG, rCB, rGX;

    uchar16 clrDiff = 0;
    uchar16 grdDiff = 0;

    if(x >= abs(d))
    {
    	rCR = vload16(offset - d, rImgR);
    	rCG = vload16(offset - d, rImgG);
        rCB = vload16(offset - d, rImgB);
    	rGX = vload16(offset - d, rGrdX);

        // three color diff
        clrDiff = (abs( lCR - rCR ) + abs( lCG - rCG ) + abs( lCB - rCB ))/(uchar)3;
        // gradient diff
        grdDiff = abs( lGX - rGX );
    }
    else
    {
        // three color diff at img boundary
        clrDiff = (abs(lCR - (uchar)UCHAR_MAX) + abs(lCG - (uchar)UCHAR_MAX) + abs(lCB - (uchar)UCHAR_MAX))/(uchar)3;
        // gradient diff
        grdDiff = abs(lGX - (uchar)UCHAR_MAX);
    }

    clrDiff = clrDiff > TAU_1 ? TAU_1 : clrDiff; 
    grdDiff = grdDiff > TAU_2 ? TAU_2 : grdDiff; 
    vstore16((uchar16)ALPHA * clrDiff + (uchar16)(1-ALPHA) * grdDiff, costVol_offset, lcostVol);


	/* *************** Right to Left Cost Volume Construction ********************** */
	// Set addresses for lImg, rImg, lGrdX and rGrdX.
    lCR = vload16(offset, rImgR);
    lCG = vload16(offset, rImgG);
    lCB = vload16(offset, rImgB);
    lGX = vload16(offset, rGrdX);

    clrDiff = 0;
    grdDiff = 0;

    if(x >= abs(d))
    {
    	rCR = vload16(offset - d, lImgR);
    	rCG = vload16(offset - d, lImgG);
        rCB = vload16(offset - d, lImgB);
    	rGX = vload16(offset - d, lGrdX);

        // three color diff
        clrDiff = (abs( lCR - rCR ) + abs( lCG - rCG ) + abs( lCB - rCB ))/(uchar)3;
        // gradient diff
        grdDiff = abs( lGX - rGX );
    }
    else
    {
        // three color diff at img boundary
        clrDiff = (abs(lCR - (uchar)UCHAR_MAX) + abs(lCG - (uchar)UCHAR_MAX) + abs(lCB - (uchar)UCHAR_MAX))/(uchar)3;
        // gradient diff
        grdDiff = abs(lGX - (uchar)UCHAR_MAX);
    }

    clrDiff = clrDiff > TAU_1 ? TAU_1 : clrDiff; 
    grdDiff = grdDiff > TAU_2 ? TAU_2 : grdDiff; 
    vstore16((uchar16)ALPHA * clrDiff + (uchar16)(1-ALPHA) * grdDiff, costVol_offset, rcostVol);
}
