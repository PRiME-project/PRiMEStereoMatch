/*---------------------------------------------------------------------------
   main.cpp - central Disparity Estimation Code
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "common.h"
#include "StereoMatch.h"

//Functions in main
int getmicinfo(void);
void *getDepthMap(void*);
void HCI(void);

//Global variables
StereoMatch *sm;
bool end_de = false;
int nOpenCLDev = 0;

int main(int argc, char** argv)
{
    //#############################################################################################################
    //# Introduction and Setup - poll for OpenCL devices
    //#############################################################################################################
	nOpenCLDev = openCLdevicepoll();

	//#############################################################################################################
    //# Start Application Processes
    //#############################################################################################################
	namedWindow("InputOutput", CV_WINDOW_AUTOSIZE );
	printf("Starting Stereo Matching Application.\n");
	sm = new StereoMatch(argc, argv, nOpenCLDev);

	//pthread setup
    void *status;
	pthread_t thread_de;
	pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    //Launch app interaction thread
    pthread_create(&thread_de, &attr, getDepthMap, NULL);

	//User interface function
	HCI();

    //#############################################################################################################
    //# Clean up threads once quit signal called
    //#############################################################################################################
	end_de = true;
	printf("MAIN: Quit signal sent\n");
    pthread_join(thread_de, &status);

	delete sm;
	printf("MAIN: Disparity Estimation Halted\n");
    return 0;
}

void *getDepthMap(void *arg)
{
	while(!end_de)
	{
		sm->Compute();
		//printf("MAIN: DE Computed...\n");
		//printf("MAIN: Press h for help text.\n\n");
	}
	return(void*)0;
}


void HCI(void)
{
	//User interface input handler
    char key = ' ';
    while(key != 'q'){
        switch(key){
            case 'h':
            {
                printf("|-------------------------------------------------------------------|\n");
                printf("| Input Options:                                                    |\n");
                printf("| h: Display this help text.                                        |\n");
                printf("| q: Quit.                                                          |\n");
                printf("|-------------------------------------------------------------------|\n");
                printf("| Control Options:                                                  |\n");
                printf("|   1-8: Change thread/core number.                                 |\n");
                printf("|   m: Switch computation mode OpenCL <-> pthreads.                 |\n");
                printf("|   f: Toggle Cost Volume Filtering (ON/OFF).                       |\n");
                printf("|-------------------------------------------------------------------|\n");
                printf("| Current Options:\n");
                printf("|   Matching Algorithm: %s\n", sm->MatchingAlgorithm ? "STEREO_GIF" : "STEREO_SGBM");
                printf("|   Computation mode: %s\n", sm->de_mode ? "OpenCL" : "pthreads");
                printf("|-------------------------------------------------------------------|\n");
                break;
            }
            case 'm':
            {
				if(sm->MatchingAlgorithm == STEREO_GIF){
					if(nOpenCLDev){
						sm->de_mode = sm->de_mode ? OCV_DE : OCL_DE;
						printf("| m: Mode changed to %s |\n", sm->de_mode ? "OpenCL on the GPU" : "C++ & pthreads on the CPU");
					}
					else{
						printf("| m: Platform must contain an OpenCL compatible device to use OpenCL Mode.\n");
					}
				}
				else{
					printf("| m: Mode can only be changed when using the STEREO_GIF Matching Algoritm.\n");
				}
				break;
            }
        }
        key = waitKey(5);
    }
    return;
}
