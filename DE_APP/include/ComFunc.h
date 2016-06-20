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
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <CL/cl_ext.h>

//OpenCV Header
#include <opencv2/opencv.hpp>

//Algorithm Definitions
#define STEREO_SGBM 0
#define STEREO_GIF  1

#define OCV_DE 0
#define OCL_DE 1

#define MAX_CPU_THREADS 8
#define MIN_CPU_THREADS 1

#define MAX_GPU_THREADS 256
#define MIN_GPU_THREADS 4

#define OCL_STATS 0

using namespace cv;
using namespace std;

typedef unsigned char   uch;
typedef unsigned short  ush;
typedef unsigned long   ulg;
