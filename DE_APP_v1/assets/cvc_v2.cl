/*---------------------------------------------------------------------------
   cvc.cl - OpenCL Cost Volume Construction Kernel
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
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
    const int x = get_global_id(0) * 20;
    const int y = get_global_id(1);
    const int d = get_global_id(2);

    /* Offset calculates the position in the linear data for the row and the column. */
    int img_offset = y * width + x;
    const int costVol_offset = ((d * height) + y) * width + x;

///	/* *************** Left to Right Cost Volume Construction ********************** */
    // Set addresses for lImg, rImg, lGrdX and rGrdX.
    float16 rC1, lC1 = vload16(0, lImg + (img_offset * 3));
    float16 rC2, lC2 = vload16(0, lImg + (img_offset * 3) + 16);
    float16 rC3, lC3 = vload16(0, lImg + (img_offset * 3) + 32);
    float16 rC4, lC4 = vload16(0, lImg + (img_offset * 3) + 48);

    float4 rGX1, lGX1 = vload4(0, lGrdX + img_offset);
    float4 rGX2, lGX2 = vload4(0, lGrdX + img_offset + 4);
    float4 rGX3, lGX3 = vload4(0, lGrdX + img_offset + 8);
    float4 rGX4, lGX4 = vload4(0, lGrdX + img_offset + 12);
    float4 rGX5, lGX5 = vload4(0, lGrdX + img_offset + 16);

    float16 clrDiff1 = 0;
    float4 clrDiff2 = 0;
    float16 grdDiff1 = 0;
    float4 grdDiff2 = 0;

    if(x >= abs(d))
    {
		img_offset = y * width + x - d;

    	rC1 = vload16(0, rImg + (img_offset * 3));
    	rC2 = vload16(0, rImg + (img_offset * 3) + 16);
    	rC3 = vload16(0, rImg + (img_offset * 3) + 32);
    	rC4 = vload16(0, rImg + (img_offset * 3) + 48);

    	rGX1 = vload4(0, rGrdX + img_offset);
    	rGX2 = vload4(0, rGrdX + img_offset + 4);
    	rGX3 = vload4(0, rGrdX + img_offset + 8);
    	rGX4 = vload4(0, rGrdX + img_offset + 12);
    	rGX5 = vload4(0, rGrdX + img_offset + 16);

        // three color diff
        lC1 = fabs(lC1 - rC1);
        lC2 = fabs(lC2 - rC2);
        lC3 = fabs(lC3 - rC3);
        lC4 = fabs(lC4 - rC4);
        // gradient diff
        grdDiff1 = (float16)(fabs( lGX1 - rGX1 ),fabs( lGX2 - rGX2 ),fabs( lGX3 - rGX3 ),fabs( lGX4 - rGX4 ));
        grdDiff2 = fabs( lGX5 - rGX5 );
    }
    else
    {
        // three color diff at img boundary
        lC1 = fabs( lC1 - 1.0); //BORDER_CONSTANT = 1.0
        lC2 = fabs( lC2 - 1.0);
        lC3 = fabs( lC3 - 1.0);
        lC4 = fabs( lC4 - 1.0);
        // gradient diff
        grdDiff1 = (float16)(fabs( lGX1 - 1.0 ), fabs( lGX2 - 1.0 ), fabs( lGX3 - 1.0 ), fabs( lGX4 - 1.0 ));
        grdDiff2 = fabs( lGX5 - 1.0 ); //BORDER_CONSTANT = 1.0
    }
    clrDiff1 = (float16)(lC1.s0 + lC1.s1 + lC1.s2,
						lC1.s3 + lC1.s4 + lC1.s5,
						lC1.s6 + lC1.s7 + lC1.s8,
						lC1.s9 + lC1.sa + lC1.sb,

						lC1.sc + lC1.sd + lC1.se,
						lC1.sf + lC2.s0 + lC2.s1,
						lC2.s2 + lC2.s3 + lC2.s4,
						lC2.s5 + lC2.s6 + lC2.s7,

						lC2.s8 + lC2.s9 + lC2.sa,
						lC2.sb + lC2.sc + lC2.sd,
						lC2.se + lC2.sf + lC3.s0,
						lC3.s1 + lC3.s2 + lC3.s3,

						lC3.s4 + lC3.s5 + lC3.s6,
						lC3.s7 + lC3.s8 + lC3.s9,
						lC3.sa + lC3.sb + lC3.sc,
						lC3.sd + lC3.se + lC3.sf) * 0.3333333333;
	
	clrDiff2 = (float4)(lC4.s0 + lC4.s1 + lC4.s2,
						lC4.s3 + lC4.s4 + lC4.s5,
						lC4.s6 + lC4.s7 + lC4.s8,
						lC4.s9 + lC4.sa + lC4.sb) * 0.3333333333;



    clrDiff1 = clrDiff1 > 0.028 ? 0.028 : clrDiff1;
    clrDiff2 = clrDiff2 > 0.028 ? 0.028 : clrDiff2;
    grdDiff1 = grdDiff1 > 0.008 ? 0.008 : grdDiff1;
    grdDiff2 = grdDiff2 > 0.008 ? 0.008 : grdDiff2;

    vstore16(0.9 * clrDiff1 + 0.1 * grdDiff1, 0, lcostVol + costVol_offset); //data, offset, addr
    vstore4 (0.9 * clrDiff2 + 0.1 * grdDiff2, 0, lcostVol + costVol_offset + 16); //data, offset, addr

///	/* *************** Right to Left Cost Volume Construction ********************** */
	// Set addresses for lImg, rImg, lGrdX and rGrdX.
    img_offset = y * width + x;

    lC1 = vload16(0, rImg + (img_offset * 3));
    lC2 = vload16(0, rImg + (img_offset * 3) + 16);
    lC3 = vload16(0, rImg + (img_offset * 3) + 32);
    lC4 = vload16(0, rImg + (img_offset * 3) + 48);

    lGX1 = vload4(0, rGrdX + img_offset);
    lGX2 = vload4(0, rGrdX + img_offset + 4);
    lGX3 = vload4(0, rGrdX + img_offset + 8);
    lGX4 = vload4(0, rGrdX + img_offset + 12);
    lGX5 = vload4(0, rGrdX + img_offset + 16);

    //grdDiff = 0;

    if(x >= abs(d))
    {
		img_offset = y * width + x + d;
    	rC1  = vload16(0, lImg + (img_offset * 3));
    	rC2  = vload16(0, lImg + (img_offset * 3) + 16);
    	rC3  = vload16(0, lImg + (img_offset * 3) + 32);
    	rC4  = vload16(0, lImg + (img_offset * 3) + 48);

    	rGX1 = vload4(0, lGrdX + img_offset);
    	rGX2 = vload4(0, lGrdX + img_offset + 4);
    	rGX3 = vload4(0, lGrdX + img_offset + 8);
    	rGX4 = vload4(0, lGrdX + img_offset + 12);
    	rGX5 = vload4(0, lGrdX + img_offset + 16);

        // three color diff
        lC1 = fabs(lC1 - rC1);
        lC2 = fabs(lC2 - rC2);
        lC3 = fabs(lC3 - rC3);
        lC4 = fabs(lC4 - rC4);
        // gradient diff
		grdDiff1 = (float16)(fabs( lGX1 - rGX1 ),fabs( lGX2 - rGX2 ),fabs( lGX3 - rGX3 ),fabs( lGX4 - rGX4 ));
        grdDiff2 = fabs( lGX5 - rGX5 );
    }
    else
    {
        // three color diff at img boundary
        lC1 = fabs( lC1 - 1.0); //BORDER_CONSTANT = 1.0
        lC2 = fabs( lC2 - 1.0);
        lC3 = fabs( lC3 - 1.0);
        lC4 = fabs( lC4 - 1.0);
        // gradient diff
        grdDiff1 = (float16)(fabs( lGX1 - 1.0 ), fabs( lGX2 - 1.0 ), fabs( lGX3 - 1.0 ), fabs( lGX4 - 1.0 ));
        grdDiff2 = fabs( lGX5 - 1.0 ); //BORDER_CONSTANT = 1.0
    }
    clrDiff1 = (float16)(lC1.s0 + lC1.s1 + lC1.s2,
						lC1.s3 + lC1.s4 + lC1.s5,
						lC1.s6 + lC1.s7 + lC1.s8,
						lC1.s9 + lC1.sa + lC1.sb,

						lC1.sc + lC1.sd + lC1.se,
						lC1.sf + lC2.s0 + lC2.s1,
						lC2.s2 + lC2.s3 + lC2.s4,
						lC2.s5 + lC2.s6 + lC2.s7,

						lC2.s8 + lC2.s9 + lC2.sa,
						lC2.sb + lC2.sc + lC2.sd,
						lC2.se + lC2.sf + lC3.s0,
						lC3.s1 + lC3.s2 + lC3.s3,

						lC3.s4 + lC3.s5 + lC3.s6,
						lC3.s7 + lC3.s8 + lC3.s9,
						lC3.sa + lC3.sb + lC3.sc,
						lC3.sd + lC3.se + lC3.sf) * 0.3333333333;
	
	clrDiff2 = (float4)(lC4.s0 + lC4.s1 + lC4.s2,
						lC4.s3 + lC4.s4 + lC4.s5,
						lC4.s6 + lC4.s7 + lC4.s8,
						lC4.s9 + lC4.sa + lC4.sb) * 0.3333333333;

    clrDiff1 = clrDiff1 > 0.028 ? 0.028 : clrDiff1;
    clrDiff2 = clrDiff2 > 0.028 ? 0.028 : clrDiff2;
    grdDiff1 = grdDiff1 > 0.008 ? 0.008 : grdDiff1;
    grdDiff2 = grdDiff2 > 0.008 ? 0.008 : grdDiff2;

    vstore16(0.9 * clrDiff1 + 0.1 * grdDiff1, 0, rcostVol + costVol_offset); //data, offset, addr
    vstore4 (0.9 * clrDiff2 + 0.1 * grdDiff2, 0, rcostVol + costVol_offset + 16); //data, offset, addr
}
