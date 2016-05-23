/*---------------------------------------------------------------------------
   main.cpp - central Disparity Estimation Code
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"
#include "StereoMatch.h"

//Functions in main
bool openCLdevicepoll(void);
void *getDepthMap(void*);
void HCI(void);

//Global variables
StereoMatch *sm;
bool end_de = false;
bool gotOpenCLDev = false;

int main(int argc, char** argv)
{
    //#############################################################################################################
    //# Introduction and Setup - poll for OpenCL devices
    //#############################################################################################################
	gotOpenCLDev = openCLdevicepoll();

	//#############################################################################################################
    //# Start Application Processes
    //#############################################################################################################
	namedWindow("InputOutput", CV_WINDOW_AUTOSIZE );
	printf("Starting Stereo Matching Application.\n");
	sm = new StereoMatch(argc, argv, gotOpenCLDev);

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
		printf("MAIN: DE Computed...\n");
		printf("MAIN: Press h for help text.\n\n");
	}
	return(void*)0;
}


void HCI(void)
{
	//User interface input handler
    char key = ' ';
    while(key != 'q'){
        if(key > '0' && key < '9')
        {
            if(!sm->de_mode){
                sm->num_threads = key - '0';
                printf("| Threading level changed to %d |\n", sm->num_threads);
            }
            else{
                printf("| Threading level can only be set when in OpenCV-CPU Mode |\n");
            }
        }
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
                printf("|   Thread level: %d\n", sm->num_threads);
                printf("|   Filtering: %s\n", sm->filter ? "ON" : "OFF");
                printf("|-------------------------------------------------------------------|\n");
                break;
            }
            case 'm':
            {
				if(sm->MatchingAlgorithm == STEREO_GIF){
					if(gotOpenCLDev){
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
            case 'f':
            {
                sm->filter = !sm->filter;
                printf("| f: Filtering turned %s |\n", sm->filter ? "ON" : "OFF");
                break;
            }
        }
        key = waitKey(5);
    }
    return;
}

bool openCLdevicepoll(void)
{
    printf("\nOpenCL Platform Information:\n");

    char* value;
    size_t valueSize;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_uint maxComputeUnits;
    cl_uint maxWorkGroupSize;
    cl_uint maxWorkItemDims;

    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);

    if(!platformCount)
    {
		printf("No OpenCL Compatible Platforms found\n");
		return false;
	}

    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

    for (int i = 0; i < (int)platformCount; i++) {

        // get all devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

        // for each device print critical attributes
        for (int j = 0; j < (int)deviceCount; j++) {

            // print device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
            printf("%d. Device: %s\n", j+1, value);
            free(value);

            // print hardware device version
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
            printf(" %d.%d Hardware version: %s\n", j+1, 1, value);
            free(value);

            // print software driver version
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
            printf(" %d.%d Software version: %s\n", j+1, 2, value);
            free(value);

            // print c version supported by compiler for device
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
            printf(" %d.%d OpenCL C version: %s\n", j+1, 3, value);
            free(value);

            // print parallel compute units
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            printf(" %d.%d Parallel compute units: %d\n", j+1, 4, maxComputeUnits);

            // print workgroup sizes
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
            printf(" %d.%d Max Work Group Size: %d\n", j+1, 5, maxWorkGroupSize);

            // print workgroup sizes
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                    sizeof(maxWorkItemDims), &maxWorkItemDims, NULL);
            printf(" %d.%d Max Work Item Dimensions: %d\n", j+1, 6, maxWorkItemDims);

            // image support?
            clGetDeviceInfo(devices[j], CL_DEVICE_IMAGE_SUPPORT,
                    sizeof(maxWorkItemDims), &maxWorkItemDims, NULL);
            printf(" %d.%d Image Support?: %s\n", j+1, 7, maxWorkItemDims ? "Yes" : "No");
        }

        free(devices);
    }

    free(platforms);
    printf("\n");

    return true;
}
