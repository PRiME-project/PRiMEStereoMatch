/*---------------------------------------------------------------------------
   ComFunc.h - Function and Header Linkage File
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sstream>
#include <cstddef>
#include <string>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <ctime>
#include <chrono>

//POSIX Threads
#include <pthread.h>

//OpenCL Header
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <CL/cl_ext.h>

//OpenCV Header
#include <opencv2/opencv.hpp>

#define BASE_DIR "../"

//Algorithm Definitions
#define STEREO_SGBM 0
#define STEREO_GIF  1

#define OCV_DE 0
#define OCL_DE 1

#define GIF_R_WIN 9
#define GIF_EPS_32F 0.0001f
#define GIF_EPS_8UC 1

#define MAX_CPU_THREADS 8
#define MIN_CPU_THREADS 1

#define MAX_GPU_THREADS 256
#define MIN_GPU_THREADS 4

#define OCL_STATS 0

using namespace cv;
using namespace std;

#ifndef COMFUNC_H
#define COMFUNC_H

enum buff_id {CVC_LIMGR, CVC_LIMGG, CVC_LIMGB, CVC_RIMGR, CVC_RIMGG, CVC_RIMGB, CVC_LGRDX, CVC_RGRDX, CV_LCV, CV_RCV, DS_LDM, DS_RDM};

#endif // COMFUNC_H
