/*---------------------------------------------------------------------------
   main.cpp - central Disparity Estimation Code
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "oclUtil.h"
#include "StereoMatch.h"
#include <chrono>
#include <thread>
#include <opencv2/highgui.hpp>

//Functions in main
void getDepthMap(StereoMatch *sm);
void HCI(StereoMatch *sm);

//Global variables
bool end_de = false;
int nOpenCLDev = 0;
int sgbm_mode = StereoSGBM::MODE_HH;

int main(int argc, const char* argv[])
{
    //#############################################################################################################
    //# Introduction and Setup - poll for OpenCL devices
    //#############################################################################################################
	nOpenCLDev = openCLdevicepoll();
#ifdef DISPLAY
	namedWindow("InputOutput", CV_WINDOW_AUTOSIZE);
#endif
	//#############################################################################################################
    //# Start Application Processes
    //#############################################################################################################
	printf("Starting Stereo Matching Application.\n");
	StereoMatch *sm = new StereoMatch(argc, argv, nOpenCLDev);
	//printf("MAIN: Press h for help text.\n\n");

	std::thread de_thread;
	//de_thread = std::thread(&StereoMatch::compute, sm, std::ref(de_time));
    de_thread = std::thread(&getDepthMap, sm);

#ifdef DISPLAY
	//User interface function
	HCI(sm);
#else
	while(1){
		std::this_thread::sleep_for (std::chrono::duration<int, std::milli>(1000));
	}
#endif
    //#############################################################################################################
    //# Clean up threads once quit signal called
    //#############################################################################################################
	end_de = true;
	printf("MAIN: Quit signal sent\n");
    de_thread.join();

	delete sm;
	printf("MAIN: Disparity Estimation Halted\n");
    return 0;
}

void getDepthMap(StereoMatch *sm)
{
	int ret = 0;
	float de_time;

	while(!(end_de || ret)){
		ret = sm->compute(de_time);
	}
	return;
}

static void on_trackbar_err(int value, void* ptr)
{
	printf("HCI: Error threshold set to %d.\n", value);
}

void HCI(StereoMatch *sm)
{
	//User interface input handler
    char key = ' ';
	float de_time;
	int dataset_idx = 0;

#ifdef DISPLAY
		imshow("InputOutput", sm->display_container);

		if(sm->media_mode == DE_IMAGE){
			cv::createTrackbar("Error Threshold", "InputOutput", &sm->error_threshold, 64, on_trackbar_err);
			on_trackbar_err(sm->error_threshold, (void*)4);
		}
#endif

    while(key != 'q')
    {
        switch(key)
        {
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
                printf("|   d:   Cycle between images datasets:\n");
				printf("|   d:   	Art, Books, Cones, Dolls, Laundry, Moebius, Teddy.n");
                printf("|   m:   Switch computation mode:\n");
                printf("|   m:      STEREO_GIF:  OpenCL <-> pthreads.\n");
                printf("|   m:      STEREO_SGBM: MODE_SGBM, MODE_HH, MODE_SGBM_3WAY\n");
                printf("|   -/=: Increase or decrease the error threshold\n");
                printf("|-------------------------------------------------------------------|\n");
                printf("| Current Options:\n");
                printf("|   a:   Matching Algorithm: %s\n", sm->MatchingAlgorithm ? "STEREO_GIF" : "STEREO_SGBM");
                printf("|   d:   Dataset: %s\n", sm->MatchingAlgorithm ? "STEREO_GIF" : "STEREO_SGBM");
                printf("|   m:   Computation mode: %s\n", sm->MatchingAlgorithm ? (sm->de_mode ? "OpenCL" : "pthreads") : (
														sgbm_mode == StereoSGBM::MODE_HH ? "MODE_HH" :
														sgbm_mode == StereoSGBM::MODE_SGBM ? "MODE_SGBM" : "MODE_SGBM_3WAY" ));
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
            case 'd':
            {
				if(sm->media_mode == DE_VIDEO){
					printf("| d: Must be in image mode to use datasets.\n");
					break;
				}
				if(sm->user_dataset){
					printf("| d: User dataset has been specified.\n");
					break;
				}
				dataset_idx = dataset_idx < dataset_names.size()-1 ? dataset_idx + 1 : 0;
				printf("| d: Dataset changed to: %s\n", dataset_names[dataset_idx].c_str());
				sm->update_dataset(dataset_names[dataset_idx]);
				break;
            }
            case 'm':
            {
				if(sm->MatchingAlgorithm == STEREO_GIF){
					if(nOpenCLDev){
						sm->de_mode = sm->de_mode ? OCV_DE : OCL_DE;
						printf("| m: STEREO_GIF Matching Algorithm:\n");
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
					printf("| m: STEREO_GIF Matching Algorithm:\n");
					printf("| m: Mode changed to %s |\n", sgbm_mode == StereoSGBM::MODE_HH ? "MODE_HH" :
															sgbm_mode == StereoSGBM::MODE_SGBM ? "MODE_SGBM" :
															"MODE_SGBM_3WAY");
				}
				break;
            }
            case 'o':
            {
            	if(sm->mask_mode == NO_MASKS){
					printf("| o: Disparity error masks not provided for the chosen dataset.\n");
					break;
            	}
				sm->mask_mode = (sm->mask_mode == MASK_NONE ? MASK_NONOCC :
								sm->mask_mode == MASK_NONOCC ? MASK_DISC :
								MASK_NONE);
				printf("| o: Disparity error mask set to: %s |\n", sm->mask_mode == MASK_NONE ? "None" :
																	sm->mask_mode == MASK_NONOCC ? "Nonocc" :
																	"Disc");
				break;
            }
            case 's':
            {
				sm->subsample_rate *= 2;
				if(sm->subsample_rate > 8)
					sm->subsample_rate = 2;
				printf("| =: Subsample rate changed to %d.\n", sm->subsample_rate);
				break;
			}
        }
        key = waitKey(1);
    }
    return;
}
