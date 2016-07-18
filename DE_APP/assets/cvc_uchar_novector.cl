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

#define ALPHA 0.9
#define TAU_1 1835 //0.028 * USHRT_MAX
#define TAU_2 524 //0.008 * USHRT_MAX

__kernel void cvc(__global const uchar* lImg,
                  __global const uchar* rImg,
                  __global const uchar* lGrdX,
                  __global const uchar* rGrdX,
                  const int height,
                  const int width,
                  __global uchar* lcostVol,
                  __global uchar* rcostVol)
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
    const int img_offset = y * width;
    const int costVol_offset = ((d * height) + y) * width;

	/* *************** Left to Right Cost Volume Construction ********************** */
    // Set addresses for lImg, rImg, lGrdX and rGrdX.
    uchar lCR  = *(lImg + (img_offset * 3) + (3 * x));
    uchar lCG  = *(lImg + (img_offset * 3) + (3 * x) + 1);
    uchar lCB  = *(lImg + (img_offset * 3) + (3 * x) + 2);
    uchar lGX  = *(lGrdX + img_offset + x);
	uchar rCR, rCG, rCB, rGX;

    ushort clrDiff = 0;
    ushort grdDiff = 0;

    if(x >= abs(d))
    {
    	rCR = *(rImg + (img_offset * 3) + 3 * (x - d));
    	rCG = *(rImg + (img_offset * 3) + 3 * (x - d) + 1);
        rCB = *(rImg + (img_offset * 3) + 3 * (x - d) + 2);
    	rGX  = *(rGrdX + img_offset + (x - d));

        // three color diff
        clrDiff += abs( lCR - rCR );
        clrDiff += abs( lCG - rCG );
        clrDiff += abs( lCB - rCB );
        clrDiff /= 3;
        // gradient diff
        grdDiff = abs( lGX - rGX );
    }
    else
    {
        // three color diff at img boundary
        clrDiff += abs(lCR - UCHAR_MAX);
        clrDiff += abs(lCG - UCHAR_MAX);
        clrDiff += abs(lCB - UCHAR_MAX);
        clrDiff /= 3;
        // gradient diff
        grdDiff = abs(lGX - UCHAR_MAX);
    }

    clrDiff = clrDiff > TAU_1 ? TAU_1 : clrDiff; 
    grdDiff = grdDiff > TAU_2 ? TAU_2 : grdDiff; 
    *(lcostVol + costVol_offset + x) = ALPHA * clrDiff + (1-ALPHA) * grdDiff;


	/* *************** Right to Left Cost Volume Construction ********************** */
	// Set addresses for lImg, rImg, lGrdX and rGrdX.
    lCR  = *(rImg + (img_offset * 3) + (3 * x));
    lCG  = *(rImg + (img_offset * 3) + (3 * x) + 1);
    lCB  = *(rImg + (img_offset * 3) + (3 * x) + 2);
    lGX  = *(rGrdX + img_offset + x);

    clrDiff = 0;
    grdDiff = 0;

    if(x >= abs(d))
    {
    	rCR = *(lImg + (img_offset * 3) + 3 * (x + d));
    	rCG = *(lImg + (img_offset * 3) + 3 * (x + d) + 1);
        rCB = *(lImg + (img_offset * 3) + 3 * (x + d) + 2);
    	rGX = *(lGrdX + img_offset + (x + d));

        // three color diff
        clrDiff += abs( lCR - rCR );
        clrDiff += abs( lCG - rCG );
        clrDiff += abs( lCB - rCB );
        clrDiff /= 3;
        // gradient diff
        grdDiff = abs( lGX - rGX );
    }
    else
    {
        // three color diff at img boundary
        clrDiff +=  abs(lCR - UCHAR_MAX);
        clrDiff +=  abs(lCG - UCHAR_MAX);
        clrDiff +=  abs(lCB - UCHAR_MAX);
        clrDiff /= 3;
        // gradient diff
        grdDiff = abs(lGX - UCHAR_MAX);
    }

    clrDiff = clrDiff > TAU_1 ? TAU_1 : clrDiff; 
    grdDiff = grdDiff > TAU_2 ? TAU_2 : grdDiff; 
    *(rcostVol + costVol_offset + x) = ALPHA * clrDiff + (1-ALPHA) * grdDiff;
}
