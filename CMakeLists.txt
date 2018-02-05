if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "DO NOT BUILD in-tree.")
endif()

cmake_minimum_required (VERSION 3.2)
project (PRiMEStereoMatch)

include_directories(include)
file(GLOB SOURCES "src/*.cpp")

add_executable(PRiMEStereoMatch ${SOURCES})
set_property(TARGET PRiMEStereoMatch PROPERTY CXX_STANDARD 11)

#pthread libraries
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
include_directories(${THREAD_INCLUDE_DIRS})

#OpenCV libraries
find_package(OpenCV REQUIRED COMPONENTS core calib3d imgproc highgui objdetect video)
include_directories(${OpenCV_INCLUDE_DIRS})

#OpenCL libraries
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

#OpenMP libraries
find_package(OpenMP REQUIRED)
include_directories(${OpenMP_INCLUDE_DIRS})
if(OPENMP_FOUND)
	message("Found OpenMP, setting Flags.")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OPENMP_FOUND)

target_link_libraries(PRiMEStereoMatch ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS} ${OpenCL_LIBRARIES} ${OpenMP_LIBRARIES})
