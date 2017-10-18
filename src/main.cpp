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
	namedWindow("InputOutput", CV_WINDOW_AUTOSIZE);
	printf("Starting Stereo Matching Application.\n");
	sm = new StereoMatch(argc, argv, nOpenCLDev);

	printf("MAIN: Press h for help text.\n\n");

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
	int ret = 0;
	float de_time;

	while(!(end_de || ret))
	{

		ret = sm->Compute(de_time);
		//printf("MAIN: DE Computed...\n");
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
                printf("| Input Options:\n");
                printf("| h: Display this help text.\n");
                printf("| q: Quit.\n");
                printf("|-------------------------------------------------------------------|\n");
                printf("| Control Options:\n");
                printf("|   1-8: Change thread/core number.\n");
                printf("|   a:   Switch matching algorithm: STEREO_GIF, STEREO_SGBM\n");
                printf("|   m:   Switch computation mode:\n");
                printf("|   m:      STEREO_GIF:  OpenCL <-> pthreads.\n");
                printf("|   m:      STEREO_SGBM: MODE_SGBM, MODE_HH, MODE_SGBM_3WAY\n");
                printf("|   t:   Switch data type: float 32 bit <-> unsigned char 8bit\n");
                printf("|   -/=: Increase or decrease the error threshold\n");
                printf("|-------------------------------------------------------------------|\n");
                printf("| Current Options:\n");
                printf("|   a:   Matching Algorithm: %s\n", sm->MatchingAlgorithm ? "STEREO_GIF" : "STEREO_SGBM");
                printf("|   m:   Computation mode: %s\n", sm->MatchingAlgorithm ? (sm->de_mode ? "OpenCL" : "pthreads") : (
														sgbm_mode == StereoSGBM::MODE_HH ? "MODE_HH" :
														sgbm_mode == StereoSGBM::MODE_SGBM ? "MODE_SGBM" : "MODE_SGBM_3WAY" ));
                printf("|   t:   Data type: %s\n", sm->imgType ? "CV_32F" : "CV_8U");
                printf("|   -/=: Error Threshold: %d\n", sm->error_threshold);
                printf("|-------------------------------------------------------------------|\n");
                break;
            }
            case 'a':
            {
				sm->MatchingAlgorithm = sm->MatchingAlgorithm ? STEREO_SGBM : STEREO_GIF;
				printf("| a: Matching Algorithm Changed to: %s |\n", sm->MatchingAlgorithm ? "STEREO_GIF" : "STEREO_SGBM");
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
				if(sm->imgType == CV_32F){
					sm->imgType = CV_8U;
					printf("| p: STEREO_GIF algorithm data type = %s |\n", "CV_8U");
				} else {
					sm->imgType = CV_32F;
					printf("| p: STEREO_GIF algorithm data type = %s |\n", "CV_32F");
				}
				if(sm->MatchingAlgorithm != STEREO_GIF){
					printf("| p: Data type only affects the STEREO_GIF algorithm.\n");
				}
				break;
            }
            case '=':
            {
				sm->error_threshold++;
				printf("| =: Error threshold increased to %d.\n", sm->error_threshold);
				break;
			}
            case '-':
            {
				if(sm->error_threshold > 0){
					sm->error_threshold--;
				printf("| -: Error threshold decreased to %d.\n", sm->error_threshold);
				}
				else
					printf("| -: Cannot decrease error threshold below 0.\n");
				break;
			}
        }
        key = waitKey(5);
    }
    return;
}
