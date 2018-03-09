if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "DO NOT BUILD in-tree.")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

cmake_minimum_required (VERSION 3.2)
project (PRiMEStereoMatch)

include_directories(include)
file(GLOB SOURCES "src/*.cpp")

add_executable(PRiMEStereoMatch ${SOURCES})
set_property(TARGET PRiMEStereoMatch PROPERTY CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#pthread libraries
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
if(Threads_FOUND)
	include_directories(${THREAD_INCLUDE_DIRS})
else(Threads_FOUND)
	message(FATAL_ERROR ">> Some form of threading library is requried for compilation, preferably PThreads.")
endif(Threads_FOUND)

#OpenCV libraries
find_package(OpenCV REQUIRED COMPONENTS core calib3d imgproc highgui video)
if(OpenCV_FOUND)
	include_directories(${OpenCV_INCLUDE_DIRS})
else(OpenCV_FOUND)
	message(">> OpenCV Required")
endif(OpenCV_FOUND)

#OpenCL libraries
find_package(OpenCL REQUIRED)
if(OpenCL_FOUND)
	include_directories(${OpenCL_INCLUDE_DIRS})
else(OpenCL_FOUND)
	message(">> OpenCL not found. OpenCL is required for GPGPU/FPGA/Accelerator use.")
endif(OpenCL_FOUND)
	
#OpenMP libraries
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
	include_directories(${OpenMP_INCLUDE_DIRS})
	message("Found OpenMP, setting Flags.")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else(OpenMP_FOUND)
	message(">> OpenMP not found. OpenMP is required for multi-threading of some intermediate processing.")
endif(OpenMP_FOUND)

#Can still compile without OpenCL
if(OpenCV_FOUND AND OpenMP_FOUND)
	target_link_libraries(PRiMEStereoMatch ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS} ${OpenCL_LIBRARIES} 	${OpenMP_LIBRARIES})
else(OpenCV_FOUND AND OpenMP_FOUND)
	message(FATAL_ERROR ">> Some form of threading library is requried for compilation, preferably PThreads.")
endif(OpenCV_FOUND AND OpenMP_FOUND)
