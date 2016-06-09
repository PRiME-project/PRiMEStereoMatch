/*---------------------------------------------------------------------------
   cvc.cl - OpenCL Cost Volume Construction Kernel
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
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
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int d = get_global_id(2);

    /* Offset calculates the position in the linear data for the row and the column. */
    const int img_offset = y * width;
    const int costVol_offset = ((d * height) + y) * width;

	/* *************** Left to Right Cost Volume Construction ********************** */
    // Set addresses for lImg, rImg, lGrdX and rGrdX.
    float lCR  = *(lImg + (img_offset * 3) + (3 * x));
    float lCG  = *(lImg + (img_offset * 3) + (3 * x) + 1);
    float lCB  = *(lImg + (img_offset * 3) + (3 * x) + 2);
    float lGX  = *(lGrdX + img_offset + x);
	float rCR, rCG, rCB, rGX;

    float clrDiff = 0;
    float grdDiff = 0;

    if(x >= abs(d))
    {
    	rCR = *(rImg + (img_offset * 3) + 3 * (x - d));
    	rCG = *(rImg + (img_offset * 3) + 3 * (x - d) + 1);
        rCB = *(rImg + (img_offset * 3) + 3 * (x - d) + 2);
    	rGX  = *(rGrdX + img_offset + (x - d));

        // three color diff
        clrDiff += fabs( lCR - rCR );
        clrDiff += fabs( lCG - rCG );
        clrDiff += fabs( lCB - rCB );
        clrDiff *= 0.3333333333f;
        // gradient diff
        grdDiff = fabs( lGX - rGX );
    }
    else
    {
        // three color diff at img boundary
        clrDiff += fabs(lCR - 1.0f); //BORDER_CONSTANT = 1.0
        clrDiff += fabs(lCG - 1.0f); //BORDER_CONSTANT = 1.0
        clrDiff += fabs(lCB - 1.0f); //BORDER_CONSTANT = 1.0
        clrDiff *= 0.3333333333f;
        // gradient diff
        grdDiff = fabs( lGX - 1.0f ); //BORDER_CONSTANT = 1.0
    }

    clrDiff = clrDiff > 0.028f ? 0.028f : clrDiff; 
    grdDiff = grdDiff > 0.008f ? 0.008f : grdDiff; 
    *(lcostVol + costVol_offset + x) = 0.9f * clrDiff + 0.1f * grdDiff;


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
        clrDiff += fabs( lCR - rCR );
        clrDiff += fabs( lCG - rCG );
        clrDiff += fabs( lCB - rCB );
        clrDiff *= 0.3333333333f;
        // gradient diff
        grdDiff = fabs( lGX - rGX );
    }
    else
    {
        // three color diff at img boundary
        clrDiff += fabs(lCR - 1.0f); //BORDER_CONSTANT = 1.0
        clrDiff += fabs(lCG - 1.0f); //BORDER_CONSTANT = 1.0
        clrDiff += fabs(lCB - 1.0f); //BORDER_CONSTANT = 1.0
        clrDiff *= 0.3333333333f;
        // gradient diff
        grdDiff = fabs( lGX - 1.0f ); //BORDER_CONSTANT = 1.0
    }

    clrDiff = clrDiff > 0.028f ? 0.028f : clrDiff; 
    grdDiff = grdDiff > 0.008f ? 0.008f : grdDiff; 
    *(rcostVol + costVol_offset + x) = 0.9f * clrDiff + 0.1f * grdDiff;
}
