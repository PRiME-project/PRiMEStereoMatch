if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "DO NOT BUILD in-tree.")
endif()

cmake_minimum_required (VERSION 3.5.1)
project (DE_APP)

include_directories(include)
file(GLOB SOURCES "src/*.cpp" "main.cpp")

add_executable(DE_APP ${SOURCES})
set_property(TARGET DE_APP PROPERTY CXX_STANDARD 11)

#pthread libraries
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
include_directories(${THREAD_INCLUDE_DIRS})
target_link_libraries(DE_APP ${CMAKE_THREAD_LIBS_INIT})

#OpenCV libraries
find_package(OpenCV REQUIRED COMPONENTS core calib3d imgproc highgui objdetect video)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(DE_APP ${OpenCV_LIBS})

#OpenCL libraries
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
target_link_libraries(DE_APP ${OpenCL_LIBRARIES})
