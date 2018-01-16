/*---------------------------------------------------------------------------
   main.cpp - central Disparity Estimation Code
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "oclUtil.h"
#include "StereoMatch.h"
#include <chrono>
#include <thread>

//Functions in main
void HCI(void);

//Global variables
bool end_de = false;
int nOpenCLDev = 0;
int sgbm_mode = StereoSGBM::MODE_HH;
std::mutex dispMap_m;

int main(int argc, char** argv)
{
    //#############################################################################################################
    //# Introduction and Setup - poll for OpenCL devices
    //#############################################################################################################
	nOpenCLDev = openCLdevicepoll();
#ifdef DISPLAY
	cv::namedWindow("InputOutput", CV_WINDOW_AUTOSIZE);
#endif
	//#############################################################################################################
    //# Start Application Processes
    //#############################################################################################################
	printf("Starting Stereo Matching Application.\n");
    StereoMatch sm = StereoMatch(argc, argv, nOpenCLDev);
	//printf("MAIN: Press h for help text.\n\n");

	cv::resizeWindow("InputOutput", sm.display_container.cols, sm.display_container.rows);

	std::thread sgbm_threads[MAX_CPU_THREADS];
	std::mutex cap_m;

    for(int thread_id = 0; thread_id < MAX_CPU_THREADS; thread_id++)
    {
        sgbm_threads[thread_id] = std::thread(&StereoMatch::sgbm_thread, std::addressof(sm),  std::ref(cap_m),  std::ref(dispMap_m), std::ref(end_de));
    }

#ifdef DISPLAY
	//std::thread display_t = std::thread(&StereoMatch::display_thread, std::addressof(sm), std::ref(dispMap_m));

    char key = ' ';
    while(key != 'q'){
        cv::imshow("InputOutput", sm.display_container);
        key = cv::waitKey(1);
    }
#else
	while(1){
		std::this_thread::sleep_for (std::chrono::duration<int, std::milli>(100));
	}
#endif
    //#############################################################################################################
    //# Clean up threads once quit signal called
    //#############################################################################################################
	end_de = true;
	printf("MAIN: Quit signal sent\n");

    for(int thread_id=0; thread_id < MAX_CPU_THREADS; thread_id++)
    {
        sgbm_threads[thread_id].join();
    }

	printf("MAIN: Disparity Estimation Halted\n");
    return 0;
}
