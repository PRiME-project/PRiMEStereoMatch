/*
 * This confidential and proprietary software may be used only as
 * authorised by a licensing agreement from ARM Limited
 *   (C) COPYRIGHT 2013 ARM Limited
 *       ALL RIGHTS RESERVED
 * The entire notice above must be reproduced on all authorised
 * copies and copies may only be made to the extent permitted
 * by a licensing agreement from ARM Limited.
 */
 /*---------------------------------------------------------------------------
   oclUtil.cpp - OpenCL Utility Function Code
  ---------------------------------------------------------------------------
   Co-Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "oclUtil.h"

int openCLdevicepoll(void)
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
    long long globalMemSize;
    cl_uint imageSupport;
    //cl_device_partition_property *partition_properties;

    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);

    if(!platformCount)
    {
		printf("No OpenCL Compatible Platforms found\n");
		return 0;
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

            // print device type
            clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, valueSize, value, NULL);
            if((int)*value == CL_DEVICE_TYPE_CPU)
				printf(" %d.%d Device Type: %s\n", j+1, 0, "CPU");
			else if((int)*value == CL_DEVICE_TYPE_GPU)
				printf(" %d.%d Device Type: %s\n", j+1, 0, "GPU");
			else if((int)*value == CL_DEVICE_TYPE_ACCELERATOR)
				printf(" %d.%d Device Type: %s\n", j+1, 0, "ACCELERATOR");
			else
				printf(" %d.%d Device Type: %s\n", j+1, 0, "DEFAULT/ALL");
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

            // print workgroup sizes
            clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE,
                    sizeof(globalMemSize), &globalMemSize, NULL);
            printf(" %d.%d Max Global Memory Size: %llu\n", j+1, 7, globalMemSize);

            // image support?
            clGetDeviceInfo(devices[j], CL_DEVICE_IMAGE_SUPPORT,
                    sizeof(imageSupport), &imageSupport, NULL);
            printf(" %d.%d Image Support?: %s\n", j+1, 8, imageSupport ? "Yes" : "No");

//            clGetDeviceInfo(devices[j], CL_DEVICE_PARTITION_PROPERTIES, sizeof(partition_properties), &partition_properties, NULL);
//            printf(" %d.%d Partition Properties: %ld\n", j+1, 8, partition_properties[0]);
        }

        free(devices);
    }

    free(platforms);
    printf("\n");

    return deviceCount;
}

bool printProfilingInfo(cl_event event)
{
    cl_ulong queuedTime = 0;
    if (!checkSuccess(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queuedTime, NULL)))
    {
        std::cerr << "Retrieving CL_PROFILING_COMMAND_QUEUED OpenCL profiling information failed. " << __FILE__ << ":"<< __LINE__ << std::endl;
        return false;
    }

    cl_ulong submittedTime = 0;
    if (!checkSuccess(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submittedTime, NULL)))
    {
        std::cerr << "Retrieving CL_PROFILING_COMMAND_SUBMIT OpenCL profiling information failed. " << __FILE__ << ":"<< __LINE__ << std::endl;
        return false;
    }

    cl_ulong startTime = 0;
    if (!checkSuccess(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL)))
    {
        std::cerr << "Retrieving CL_PROFILING_COMMAND_START OpenCL profiling information failed. " << __FILE__ << ":"<< __LINE__ << std::endl;
        return false;
    }

    cl_ulong endTime = 0;
    if (!checkSuccess(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL)))
    {
        std::cerr << "Retrieving CL_PROFILING_COMMAND_END OpenCL profiling information failed. " << __FILE__ << ":"<< __LINE__ << std::endl;
        return false;
    }

    //cout << "Profiling information:\n";
    /* OpenCL returns times in nano seconds. Print out the times in milliseconds (divide by a million). */
    std::cout << "Queued time: \t" << (submittedTime - queuedTime) / 1000000.0 << "ms, ";
    std::cout << "Wait time: \t" << (startTime - submittedTime) / 1000000.0 << "ms, ";
    std::cout << "Run time: \t" << (endTime - startTime) / 1000000.0 << "ms" << std::endl;

    return true;
}

bool printSupported2DImageFormats(cl_context context)
{
    /* Get the number of supported image formats in order to allocate the correct amount of memory. */
    cl_uint numberOfImageFormats;
    if (!checkSuccess(clGetSupportedImageFormats(context, 0, CL_MEM_OBJECT_IMAGE2D, 0, NULL, &numberOfImageFormats)))
    {
        std::cerr << "Getting the number of supported 2D image formats failed. " << __FILE__ << ":"<< __LINE__ << std::endl;
        return false;
    }

    /* Get the list of supported image formats. */
    cl_image_format* imageFormats = new cl_image_format[numberOfImageFormats];
    if (!checkSuccess(clGetSupportedImageFormats(context, 0, CL_MEM_OBJECT_IMAGE3D, numberOfImageFormats, imageFormats, NULL)))
    {
        std::cerr << "Getting the list of supported 2D image formats failed. " << __FILE__ << ":"<< __LINE__ << std::endl;
        return false;
    }

    std::cout << numberOfImageFormats << " Image formats supported";

    if (numberOfImageFormats > 0)
    {
        std::cout << " (channel order, channel data type):" << std::endl;
    }
    else
    {
        std::cout << "." << std::endl;
    }

    for (unsigned int i = 0; i < numberOfImageFormats; i++)
    {
        std::cout << imageChannelOrderToString(imageFormats[i].image_channel_order) << ", " << imageChannelDataTypeToString(imageFormats[i].image_channel_data_type) << std::endl;
    }

    delete[] imageFormats;

    return true;
}

bool cleanUpOpenCL(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem* memoryObjects, int numberOfMemoryObjects)
{
    bool returnValue = true;
    if (context != 0)
    {
        if (!checkSuccess(clReleaseContext(context)))
        {
            std::cerr << "Releasing the OpenCL context failed. " << __FILE__ << ":"<< __LINE__ << std::endl;
            returnValue = false;
        }
    }

    if (commandQueue != 0)
    {
        if (!checkSuccess(clReleaseCommandQueue(commandQueue)))
        {
            std::cerr << "Releasing the OpenCL command queue failed. " << __FILE__ << ":"<< __LINE__ << std::endl;
            returnValue = false;
        }
    }

    if (kernel != 0)
    {
        if (!checkSuccess(clReleaseKernel(kernel)))
        {
            std::cerr << "Releasing the OpenCL kernel failed. " << __FILE__ << ":"<< __LINE__ << std::endl;
            returnValue = false;
        }
    }

    if (program != 0)
    {
        if (!checkSuccess(clReleaseProgram(program)))
        {
            std::cerr << "Releasing the OpenCL program failed. " << __FILE__ << ":"<< __LINE__ << std::endl;
            returnValue = false;
        }
    }

    for (int index = 0; index < numberOfMemoryObjects; index++)
    {
        if (memoryObjects[index] != 0)
        {
            if (!checkSuccess(clReleaseMemObject(memoryObjects[index])))
            {
                std::cerr << "Releasing the OpenCL memory object " << index << " failed. " << __FILE__ << ":"<< __LINE__ << std::endl;
                returnValue = false;
            }
        }
    }

    return returnValue;
}

bool createContext(cl_context* context)
{
    cl_int errorNumber = 0;
    cl_uint numberOfPlatforms = 0;
    cl_platform_id firstPlatformID = 0;

    /* Retrieve a single platform ID. */
    if (!checkSuccess(clGetPlatformIDs(1, &firstPlatformID, &numberOfPlatforms)))
    {
        std::cerr << "Retrieving OpenCL platforms failed. " << __FILE__ << ":"<< __LINE__ << std::endl;
        return false;
    }

    if (numberOfPlatforms <= 0)
    {
        std::cerr << "No OpenCL platforms found. " << __FILE__ << ":"<< __LINE__ << std::endl;
        return false;
    }

	cl_uint deviceCount;
	// get all devices
	clGetDeviceIDs(firstPlatformID, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);

	struct {cl_device_type type; const char* name; cl_uint dcount; } devices[] =
	{
        { CL_DEVICE_TYPE_CPU, "CL_DEVICE_TYPE_CPU", 0 },
        { CL_DEVICE_TYPE_GPU, "CL_DEVICE_TYPE_GPU", 0 },
        { CL_DEVICE_TYPE_ACCELERATOR, "CL_DEVICE_TYPE_ACCELERATOR", 0 }
	};

    const int NUM_OF_DEVICE_TYPES = sizeof(devices)/sizeof(devices[0]);

	printf("Number of devices available of each type:\n");
	for(int i = 0; i < NUM_OF_DEVICE_TYPES; ++i)
	{
		errorNumber = clGetDeviceIDs(firstPlatformID, devices[i].type, 0, 0, &devices[i].dcount);
		if(CL_DEVICE_NOT_FOUND == errorNumber)
		{
			devices[i].dcount = 0;
			errorNumber = CL_SUCCESS;
		}
		printf("\t%s: %d\n", devices[i].name, devices[i].dcount);
	}

    /* Get a context with a device from the platform found above. */
    cl_context_properties contextProperties [] = {CL_CONTEXT_PLATFORM, (cl_context_properties)firstPlatformID, 0};
    if(devices[NUM_OF_DEVICE_TYPES-1].dcount > 0) //choose Accelerator as first preference if present
		*context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_ACCELERATOR, context_notify, NULL, &errorNumber);
    else if(devices[NUM_OF_DEVICE_TYPES-2].dcount > 0) //choose GPU as second preference if present
		*context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, context_notify, NULL, &errorNumber);
	else //choose CPU as last preference
		*context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU, context_notify, NULL, &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        std::cerr << "Creating an OpenCL context failed. " << __FILE__ << ":"<< __LINE__ << std::endl;
        return false;
    }

    return true;
}

bool createSubDeviceContext(cl_context* context, cl_int numComputeUnits) //work in progress
{
    cl_int errorNumber = 0;
    cl_uint numberOfPlatforms = 0;
    cl_platform_id firstPlatformID = 0;

    /* Retrieve a single platform ID. */
    if (!checkSuccess(clGetPlatformIDs(1, &firstPlatformID, &numberOfPlatforms)))
    {
        std::cerr << "Retrieving OpenCL platforms failed. " << __FILE__ << ":"<< __LINE__ << std::endl;
        return false;
    }

    if (numberOfPlatforms <= 0)
    {
        std::cerr << "No OpenCL platforms found. " << __FILE__ << ":"<< __LINE__ << std::endl;
        return false;
    }

    /* Get a context with a GPU device from the platform found above. */
    cl_context_properties contextProperties [] = {CL_CONTEXT_PLATFORM, (cl_context_properties)firstPlatformID, 0};

    cl_uint numDevices;
    cl_device_id device_id;

	// Get Device ID from selected platform:
	clGetDeviceIDs( firstPlatformID, CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id, &numDevices);

	// Create two sub-device properties: Partition By Counts
	/* Examples:
		cl_device_partition_property props[] = { CL_DEVICE_PARTITION_BY_COUNTS, 4, CL_DEVICE_PARTITION_BY_COUNTS_LIST_END, 0};
		cl_device_partition_property props[] = { CL_DEVICE_PARTITION_EQUALLY, 8, 0};
	*/
	#ifdef CL_VERSION_1_1
	//cl_device_partition_property_ext pprops[] = { CL_DEVICE_PARTITION_BY_COUNTS_EXT, 4, 4, CL_PROPERTIES_LIST_END_EXT, 0};
	#else
	//cl_device_partition_property pprops[] = { CL_DEVICE_PARTITION_BY_COUNTS, 4, CL_DEVICE_PARTITION_BY_COUNTS_LIST_END, 0};
	#endif /* CL_VERSION_1_1 */



	cl_device_id subdevice_id;
	// Create the sub-devices:
	#ifdef CL_VERSION_1_1
	//if (!checkSuccess(clCreateSubDevicesEXT(device_id, pprops, 1, &subdevice_id, &numDevices)))
	#else
	//if (!checkSuccess(clCreateSubDevices(device_id, pprops, 1, &subdevice_id, &numDevices)))
	#endif /* CL_VERSION_1_1 */
    {
        std::cerr << "Creating an OpenCL Sub Device failed. " << __FILE__ << ":"<< __LINE__ << std::endl;
        return false;
    }
	// Create the context:
	*context = clCreateContext(contextProperties, 1, &subdevice_id, NULL, NULL, &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        std::cerr << "Creating an OpenCL context failed. " << __FILE__ << ":"<< __LINE__ << std::endl;
        return false;
    }

    return true;
}

bool createCommandQueue(cl_context context, cl_command_queue* commandQueue, cl_device_id* device)
{
    cl_int errorNumber = 0;
    cl_device_id* devices = NULL;
    size_t deviceBufferSize = -1;

    /* Retrieve the size of the buffer needed to contain information about the devices in this OpenCL context. */
    if (!checkSuccess(clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize)))
    {
        std::cerr << "Failed to get OpenCL context information. " << __FILE__ << ":"<< __LINE__ << std::endl;
        return false;
    }

    if(deviceBufferSize == 0)
    {
        std::cerr << "No OpenCL devices found. " << __FILE__ << ":"<< __LINE__ << std::endl;
        return false;
    }

    /* Retrieve the list of devices available in this context. */
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    if (!checkSuccess(clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL)))
    {
        std::cerr << "Failed to get the OpenCL context information. " << __FILE__ << ":"<< __LINE__ << std::endl;
        delete [] devices;
        return false;
    }

    /* Use the first available device in this context. */
    *device = devices[0];
    delete [] devices;

	//const cl_command_queue_properties queue_properties = NULL;
    /* Set up the command queue with the selected device. */
    //*commandQueue = clCreateCommandQueue(context, *device, &queue_properties, &errorNumber);
    *commandQueue = clCreateCommandQueue(context, *device, CL_QUEUE_PROFILING_ENABLE, &errorNumber);
    if (!checkSuccess(errorNumber))
    {
        std::cerr << "Failed to create the OpenCL command queue. " << __FILE__ << ":"<< __LINE__ << std::endl;
        return false;
    }

    return true;
}

bool createProgram(cl_context context, cl_device_id device, std::string filename, cl_program* program)
{
    cl_int errorNumber = 0;
    std::ifstream kernelFile(filename.c_str(), std::ios::in);

    if(!kernelFile.is_open())
    {
        std::cerr << "Unable to open " << filename << ". " << __FILE__ << ":"<< __LINE__ << std::endl;
        return false;
    }

    /*
     * Read the kernel file into an output stream.
     * Convert this into a char array for passing to OpenCL.
     */
    std::ostringstream outputStringStream;
    outputStringStream << kernelFile.rdbuf();
    std::string srcStdStr = outputStringStream.str();
    const char* charSource = srcStdStr.c_str();

    *program = clCreateProgramWithSource(context, 1, &charSource, NULL, &errorNumber);
    if (!checkSuccess(errorNumber) || program == NULL)
    {
        std::cerr << "Failed to create OpenCL program. " << __FILE__ << ":"<< __LINE__ << std::endl;
        return false;
    }

    /* Try to build the OpenCL program. */
    bool buildSuccess = checkSuccess(clBuildProgram(*program, 0, NULL, NULL, NULL, NULL));

    /* Get the size of the build log. */
    size_t logSize = 0;
    clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

    /*
     * If the build succeeds with no log, an empty string is returned (logSize = 1),
     * we only want to print the message if it has some content (logSize > 1).
     */
    if (logSize > 1)
    {
        char* log = new char[logSize];
        clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);

        std::string* stringChars = new std::string(log, logSize);
        std::cerr << "Build log:\n " << *stringChars << std::endl;

        delete[] log;
        delete stringChars;
    }

    if (!buildSuccess)
    {
        clReleaseProgram(*program);
        std::cerr << "Failed to build OpenCL program. " << __FILE__ << ":"<< __LINE__ << std::endl;
        return false;
    }

    return true;
}

//inline bool checkSuccess(cl_int errorNumber)
//{
//    if (errorNumber != CL_SUCCESS)
//    {
//        std::cerr << "OpenCL error: " << errorNumberToString(errorNumber) << std::endl;
//        return false;
//    }
//    return true;
//}

std::string imageChannelOrderToString(cl_channel_order channelOrder)
{
    switch (channelOrder)
    {
        case CL_R:
            return "CL_R";
        case CL_A:
            return "CL_A";
        case CL_RG:
             return "CL_RG";
        case CL_RA:
             return "CL_RA";
        case CL_RGB:
            return "CL_RGB";
        case CL_RGBA:
            return "CL_RGBA";
        case CL_BGRA:
            return "CL_BGRA";
        case CL_ARGB:
            return "CL_ARGB";
        case CL_INTENSITY:
            return "CL_INTENSITY";
        case CL_LUMINANCE:
            return "CL_LUMINANCE";
        case CL_Rx:
            return "CL_Rx";
        case CL_RGx:
            return "CL_RGx";
        case CL_RGBx:
            return "CL_RGBx";
        default:
            return "Unknown image channel order";
    }
}

std::string imageChannelDataTypeToString(cl_channel_type channelDataType)
{
    switch (channelDataType)
    {
        case CL_SNORM_INT8:
            return "CL_SNORM_INT8";
        case CL_SNORM_INT16:
            return "CL_SNORM_INT16";
        case CL_UNORM_INT8:
            return "CL_UNORM_INT8";
        case CL_UNORM_INT16:
            return "CL_UNORM_INT16";
        case CL_UNORM_SHORT_565:
            return "CL_UNORM_SHORT_565";
        case CL_UNORM_SHORT_555:
            return "CL_UNORM_SHORT_555";
        case CL_UNORM_INT_101010:
            return "CL_UNORM_INT_101010";
        case CL_SIGNED_INT8:
            return "CL_SIGNED_INT8";
        case CL_SIGNED_INT16:
            return "CL_SIGNED_INT16";
        case CL_SIGNED_INT32:
            return "CL_SIGNED_INT32";
        case CL_UNSIGNED_INT8:
            return "CL_UNSIGNED_INT8";
        case CL_UNSIGNED_INT16:
            return "CL_UNSIGNED_INT16";
        case CL_UNSIGNED_INT32:
            return "CL_UNSIGNED_INT32";
        case CL_HALF_FLOAT:
            return "CL_HALF_FLOAT";
        case CL_FLOAT:
            return "CL_FLOAT";
        default:
            return "Unknown image channel data type";
    }
}

std::string errorNumberToString(cl_int errorNumber)
{
    switch (errorNumber)
    {
        case CL_SUCCESS:
            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:
            return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:
            return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return "CL_MAP_FAILURE";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_INVALID_VALUE:
            return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
            return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:
            return "CL_INVALID_MIP_LEVEL";
        default:
            return "Unknown error";
    }
}

bool isExtensionSupported(cl_device_id device, std::string extension)
{
    if (extension.empty())
    {
        return false;
    }

    /* First find out how large the ouput of the OpenCL device query will be. */
    size_t extensionsReturnSize = 0;
    if (!checkSuccess(clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &extensionsReturnSize)))
    {
        std::cerr << "Failed to get return size from clGetDeviceInfo for parameter CL_DEVICE_EXTENSIONS. " << __FILE__ << ":"<< __LINE__ << std::endl;
        return false;
    }

    /* Allocate enough memory for the output. */
    char* extensions = new char[extensionsReturnSize];

    /* Get the list of all extensions supported. */
    if (!checkSuccess(clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, extensionsReturnSize, extensions, NULL)))
    {
        std::cerr << "Failed to get data from clGetDeviceInfo for parameter CL_DEVICE_EXTENSIONS. " << __FILE__ << ":"<< __LINE__ << std::endl;
        delete [] extensions;
        return false;
    }

    /* See if the requested extension is in the list. */
    std::string* extensionsString = new std::string(extensions);
    bool returnResult = false;
    if (extensionsString->find(extension) != std::string::npos)
    {
        returnResult = true;
    }

    delete [] extensions;
    delete extensionsString;

    return returnResult;
}

void context_notify(const char *notify_message, const void *private_info, size_t cb, void *user_data)
{
          printf("OpenCL Notification:\n\t%s\n", notify_message);
}
