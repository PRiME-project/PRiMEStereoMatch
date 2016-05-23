/*---------------------------------------------------------------------------
   cvc_image.cl - OpenCL Cost Volume Construction Kernel
				- **Uses OpenCL Image Types**
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

/**
 * \brief Disparity Estimation kernel function.
 * \param[in] lImg - Left Input image data.
 * \param[in] rImg - Right Input image data.
 * \param[in] lGrdX - Left Input X dim gradient data.
 * \param[in] rGrdX - Right Input X dim gradient data.
 * \param[in] height - Height of the image.
 * \param[in] width - Width of the image.
 * \param[out] lcostVol - Calculated pixel cost for Cost Volume.
 * \param[out] rcostVol - Calculated pixel cost for Cost Volume.
 */
__kernel void cvc_image(__read_only image2d_t lImg,
                  		__read_only image2d_t rImg,
		                __read_only image2d_t lGrdX,
                  		__read_only image2d_t rGrdX,
                  		const int height,
                  		const int width,
                  		__write_only image3d_t lcostVol,
                  		__write_only image3d_t rcostVol)
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

	int2 coordinate2D = (int2)(x,y);
	int4 coordinate3D = (int4)(x,y,d,0);
    /* Offset calculates the position in the linear data for the row and the column. */
    //const int img_offset = y * width;
    //const int costVol_offset = ((d * height) + y) * width;

	/* *************** Left to Right Cost Volume Construction ********************** */
    // Set addresses for lImg, rImg, lGrdX and rGrdX.
    float4 rC, lC  = read_imagef(lImg, sampler, (int2)(x,y));
    float rGX, lGX  = read_imagef(lGrdX, sampler, (int2)(x,y)).x;

    float clrDiff = 0;
    float grdDiff = 0;

    if(x >= abs(d))
    {
    	rC = read_imagef(rImg, sampler, (int2)(x-d,y));
    	rGX = read_imagef(rGrdX, sampler, (int2)(x-d,y)).x;

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
	clrDiff = (lC.s0 + lC.s1 + lC.s2) * 0.3333333333;

    clrDiff = clrDiff > 0.028 ? 0.028 : clrDiff; 
    grdDiff = grdDiff > 0.008 ? 0.008 : grdDiff;
	float cost = 0.9 * clrDiff + 0.1 * grdDiff;
	write_imagef(lcostVol, coordinate3D, cost);


	/* *************** Right to Left Cost Volume Construction ********************** */
	// Set addresses for lImg, rImg, lGrdX and rGrdX.
    lC  = read_imagef(rImg, sampler, (int2)(x,y));
    lGX  = read_imagef(rGrdX, sampler, (int2)(x,y)).x;

    if(x >= abs(d))
    {
		rC = read_imagef(lImg, sampler, (int2)(x+d,y));
    	rGX = read_imagef(lGrdX, sampler, (int2)(x+d,y)).x;

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
	clrDiff = (lC.s0 + lC.s1 + lC.s2) * 0.3333333333;

    clrDiff = clrDiff > 0.028 ? 0.028 : clrDiff; 
    grdDiff = grdDiff > 0.008 ? 0.008 : grdDiff; 
	write_imagef(rcostVol, coordinate3D, 0.9 * clrDiff + 0.1 * grdDiff);
}
