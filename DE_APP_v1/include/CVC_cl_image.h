/*---------------------------------------------------------------------------
   CVC_cl_image.h - OpenCL Cost Volume Construction Header
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "common.h"
#include "image.h"

// CVPR 11
#define BORDER_THRES 0.011764
#define BORDER_CONSTANT 1.0

//#define TAU_1 0.7
//#define TAU_2 0.2
#define TAU_1 0.028
#define TAU_2 0.008
#define ALPHA 0.9

//
// TAD + GRD for Cost Computation
//
class CVC_cli
{
public:

    Mat lImg_ref;
    //Data Variables
    Mat lGray, rGray;
	Mat lGrdX, rGrdX;
	Mat tmp;
	int maxDis;

	//OpenCL Variables
	cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    cl_kernel kernel;
    cl_device_id device;
    unsigned int numberOfMemoryObjects;
    cl_mem memoryObjects[6] = {0, 0, 0, 0, 0, 0};
    cl_int errorNumber;
    cl_event event;

    cl_int width, height, dispRange;
    cl_image_format format_color, format_grad, format_cv;
    size_t origin[3] = {0, 0, 0};
    size_t region2D[3] = {(size_t)width, (size_t)height, 1};
    size_t globalWorksize[3];

    CVC_cli(Mat l, const int d);
    ~CVC_cli(void);

	int buildCV(const Mat& lImg, const Mat& rImg, Mat* lcostVol, Mat* rcostVol);
};
