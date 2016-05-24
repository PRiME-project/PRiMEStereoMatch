/*---------------------------------------------------------------------------
   DispEst.h - Disparity Estimation Class Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "CVC.h"
#include "CVC_cl.h"
#include "CVF.h"
#include "CVF_cl.h"
#include "DispSel.h"
#include "DispSel_cl.h"
#include "PP.h"

//
// Overarching Disparity Estimation Class
//
class DispEst
{
public:

    Mat lImg;
    Mat rImg;

    int hei;
    int wid;
    int maxDis;
    int threads;
    bool useOCL;

	//CVC & CVF
    Mat* lcostVol;
    Mat* rcostVol;
    //DispSel
    Mat lDisMap;
    Mat rDisMap;
    //PP
    Mat lValid;
    Mat rValid;

    CVC* constructor;
    CVC_cl* constructor_cl;
//    CVC_cli* constructor_cl_image;
    CVF* filter;
    CVF_cl* filter_cl;
    DispSel* selector;
    DispSel_cl* selector_cl;
    PP* postProcessor;

    DispEst(Mat l, Mat r, const int d, const int t, bool ocl);
    ~DispEst(void);

    void CostConst();
    void CostConst_CPU();
    void CostConst_GPU();

    void CostFilter();
    void CostFilter_CPU();
    void CostFilter_GPU();

    void DispSelect_CPU();
    void DispSelect_GPU();

    void PostProcess_CPU();
};
