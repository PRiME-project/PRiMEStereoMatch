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
   oclUtil.h - OpenCL Utility Function Header
  ---------------------------------------------------------------------------
   Co-Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#ifndef OCLUTIL_H
#define OCLUTIL_H

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

//OpenCL Header
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <CL/cl_ext.h>

/**
 * \file common.h
 * \brief Functions to simplify the use of OpenCL API.
 */
//Added functions:
int openCLdevicepoll(void);

/**
 * \brief Convert OpenCL error numbers to their string form.
 * \details Uses the error number definitions from cl.h.
 * \param[in] errorNumber The error number returned from an OpenCL command.
 * \return A name of the error.
 */
std::string errorNumberToString(cl_int errorNumber);

/**
 * \brief Check an OpenCL error number for errors.
 * \details If errorNumber is not CL_SUCESS, the function will print the string form of the error number.
 * \param[in] errorNumber The error number returned from an OpenCL command.
 * \return False if errorNumber != CL_SUCCESS, true otherwise.
 */
inline bool checkSuccess(cl_int errorNumber)
{
    if (errorNumber != CL_SUCCESS)
    {
        std::cerr << "OpenCL error: " << errorNumberToString(errorNumber) << std::endl;
        return false;
    }
    return true;
}

/**
 * \brief Print the profiling information associated with an OpenCL event.
 * \details Prints the time spent in the command queue, the time spent waiting before being submitted to a device, and the execution time.
 * \param[in] event The event to get profiling information for.
 * \return False if an error occurred, otherwise true.
 */
bool printProfilingInfo(cl_event event);

/**
 * \brief Print a list of the 2D OpenCL image formats supported.
 * \param[in] context The OpenCL context to use.
 * \return False if an error occurred, otherwise true.
 */
bool printSupported2DImageFormats(cl_context context);

/**
 * \brief Convert cl_channel_order values into their string form.
 * \details Uses the channel order definitions from cl.h.
 * \param[in] channelOrder The channel order value to convert.
 * \return The string form of the channel order.
 */
std::string imageChannelOrderToString(cl_channel_order channelOrder);

/**
 * \brief Convert cl_channel_type values into their string form.
 * \details Uses the channel data type definitions from cl.h.
 * \param[in] channelDataType The channel data type value to convert.
 * \return The string form of the channel data type.
 */
std::string imageChannelDataTypeToString(cl_channel_type channelDataType);

/**
 * \brief Release any OpenCL objects that have been created.
 * \details If any of the OpenCL objects passed in are not NULL, they will be freed using the appropriate OpenCL function.
 * \return False if an error occurred, otherwise true.
 */
bool cleanUpOpenCL(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem* memoryObjects, int numberOfMemoryObjects);

/**
 * \brief Create an OpenCL context on a device on the first available platform.
 * \param[out] context Pointer to the created OpenCL context.
 * \return False if an error occurred, otherwise true.
 */
bool createContext(cl_context* context);

/**
 * \brief Create an OpenCL context for a sub-device of a device on the first available platform.
 * \param[out] context Pointer to the created OpenCL context.
 * \return False if an error occurred, otherwise true.
 */
bool createSubDeviceContext(cl_context* context, cl_int numComputeUnits);

/**
 * \brief Create an OpenCL command queue for a given context.
 * \param[in] context The OpenCL context to use.
 * \param[out] commandQueue The created OpenCL command queue.
 * \param[out] device The device in which the command queue is created.
 * \return False if an error occurred, otherwise true.
 */
bool createCommandQueue(cl_context context, cl_command_queue* commandQueue, cl_device_id* device);

/**
 * \brief Create an OpenCL program from a given file and compile it.
 * \param[in] context The OpenCL context in use.
 * \param[in] device The OpenCL device to compile the kernel for.
 * \param[in] filename Name of the file containing the OpenCL kernel code to load.
 * \param[out] program The created OpenCL program object.
 * \return False if an error occurred, otherwise true.
 */
bool createProgram(cl_context context, cl_device_id device, std::string filename, cl_program* program);

/**
 * \brief Query an OpenCL device to see if it supports an extension.
 * \param[in] device The device to query.
 * \param[in] extension The string name of the extension to query for.
 * \return True if the extension is supported on the given device, false otherwise.
 */
bool isExtensionSupported(cl_device_id device, std::string extension);

void context_notify(const char *notify_message, const void *private_info, size_t cb, void *user_data);

#endif
