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
void *getDepthMap(void*);
void HCI(void);

//Global variables
StereoMatch *sm;
bool end_de = false;
int nOpenCLDev = 0;
int imgType = CV_32F;
int sgbm_mode = StereoSGBM::MODE_HH;

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
		if(imgType != sm->imgType)
			sm->imgTypeChange(imgType);
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
                printf("|   t: Switch data type float 32 bit <-> unsigned char 8bit.        |\n");
                printf("|-------------------------------------------------------------------|\n");
                printf("| Current Options:                                                  |\n");
                printf("|   Matching Algorithm: %s\n", sm->MatchingAlgorithm ? "STEREO_GIF" : "STEREO_SGBM");
                printf("|   Computation mode: %s\n", sm->de_mode ? "OpenCL" : "pthreads");
                printf("|   Type mode: %s\n", sm->imgType ? "CV_32F" : "CV_8U");
                printf("|-------------------------------------------------------------------|\n");
                break;
            }
            case 'm':
            {
				if(sm->MatchingAlgorithm == STEREO_GIF){
					if(nOpenCLDev){
						sm->de_mode = sm->de_mode ? OCV_DE : OCL_DE;
						printf("| m: STEREO_GIF Matching Algoritm:\n");
						printf("| m: Mode changed to %s |\n", sm->de_mode ? "OpenCL on the GPU" : "C++ & pthreads on the CPU");
					}
					else{
						printf("| m: Platform must contain an OpenCL compatible device to use OpenCL Mode.\n");
					}
				}
				else if(sm->MatchingAlgorithm == STEREO_SGBM){
					sgbm_mode = (sgbm_mode == StereoSGBM::MODE_HH ? StereoSGBM::MODE_SGBM :
								sgbm_mode == StereoSGBM::MODE_SGBM ? StereoSGBM::MODE_SGBM_3WAY :
								StereoSGBM::MODE_HH);
					sm->ssgbm->setMode(sgbm_mode);
					printf("| m: STEREO_GIF Matching Algoritm:\n");
					printf("| m: Mode changed to %s |\n", sgbm_mode == StereoSGBM::MODE_HH ? "MODE_HH" :
															sgbm_mode == StereoSGBM::MODE_SGBM ? "MODE_SGBM" :
															"MODE_SGBM_3WAY");
				}
				break;
            }
            case 't':
            {
				if(sm->MatchingAlgorithm == STEREO_GIF)
				{
					if(imgType == CV_32F){
						imgType = CV_8U;
						printf("| p: STEREO_GIF Algorithm Image Type = %s |\n", "CV_8U");
					}
					else {
						imgType = CV_32F;
						printf("| p: STEREO_GIF algorithm image type = %s |\n", "CV_32F");
					}
				}
				else{
					printf("| p: Must be using the STEREO_GIF algorithm to change image type.\n");
				}
				break;
            }
        }
        key = waitKey(5);
    }
    return;
}
