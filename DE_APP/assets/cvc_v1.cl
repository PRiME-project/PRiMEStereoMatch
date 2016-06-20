/*---------------------------------------------------------------------------
   cvc.cl - OpenCL Cost Volume Construction Kernel
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/

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
__kernel void cvc(__global const float* restrict lImg,
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

    if(x >= abs(d))
    {
		img_offset = y * width + x - d;
    	rC = vload16(0, rImg + (img_offset * 3));
    	rGX = vload4(0, rGrdX + img_offset);

        // three color diff
        lC = fabs(lC - rC);
        // gradient diff
        grdDiff = fabs( lGX - rGX );
    }
    else
    {
        // three color diff at img boundary
        lC = fabs( lC - 1.0); //BORDER_CONSTANT = 1.0
        // gradient diff
        grdDiff = fabs( lGX - 1.0 ); //BORDER_CONSTANT = 1.0
    }
    clrDiff = (float4)(lC.s0 + lC.s1 + lC.s2,
					lC.s3 + lC.s4 + lC.s5,
					lC.s6 + lC.s7 + lC.s8,
					lC.s9 + lC.sa + lC.sb) * 0.3333333333;

    clrDiff = clrDiff > 0.028 ? 0.028 : clrDiff;
    grdDiff = grdDiff > 0.008 ? 0.008 : grdDiff;
    vstore4(0.9 * clrDiff + 0.1 * grdDiff, 0, lcostVol + costVol_offset); //data, offset, addr
//	*(lcostVol + costVol_offset + 4) = 0.9 * clrDiff.s4 + 0.1 * grdDiff.s4;


///	/* *************** Right to Left Cost Volume Construction ********************** */
	// Set addresses for lImg, rImg, lGrdX and rGrdX.
    img_offset = y * width + x;
    lC  = vload16(0, rImg + (img_offset * 3));
    lGX = vload4(0, rGrdX + img_offset);

    grdDiff = 0;

    if(x >= abs(d))
    {
		img_offset = y * width + x + d;
    	rC  = vload16(0, lImg + (img_offset * 3));
    	rGX = vload4(0, lGrdX + img_offset);

        // three color diff
        lC = fabs(lC - rC);
        // gradient diff
        grdDiff = fabs( lGX - rGX );
    }
    else
    {
        // three color diff at img boundary
        lC = fabs( lC - 1.0); //BORDER_CONSTANT = 1.0
        // gradient diff
        grdDiff = fabs( lGX - 1.0 ); //BORDER_CONSTANT = 1.0
    }
    clrDiff = (float4)(lC.s0 + lC.s1 + lC.s2,
					lC.s3 + lC.s4 + lC.s5,
					lC.s6 + lC.s7 + lC.s8,
					lC.s9 + lC.sa + lC.sb) * 0.3333333333;

    clrDiff = clrDiff > 0.028 ? 0.028 : clrDiff;
    grdDiff = grdDiff > 0.008 ? 0.008 : grdDiff;
	vstore4(0.9 * clrDiff + 0.1 * grdDiff, 0, rcostVol + costVol_offset); //data, offset, addr
	//*(rcostVol + costVol_offset + 4) = 0.9 * clrDiff.s4 + 0.1 * grdDiff.s4;
}
