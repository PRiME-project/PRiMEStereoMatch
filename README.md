# PRiMEStereoMatch

<p align="center">
<img src="docs/de_examples.png" alt="Examples Image Pairs" width=80%>
</p>

## Theoretical Background

A heterogeneous and fully parallel stereo matching algorithm for depth estimation. Stereo Matching is based on the disparity estimation algorithm, an algorithm designed to calculate 3D depth information about a scene from a pair of 2D images captured by a stereoscopic camera. The algorithm contains the following stages:

* Cost Volume Construction - weighted absolute difference of colours and gradients function.
* Cost Volume Filtering - local adaptive support weight (ADSW) Guided Image Filter (GIF) function.  
* Disparity Selection - winner-takes-all minimum cost search and corresponding disparity selection.  
* Post Processing - left-right occlusion check, invalid pixel removal and weight-median filtering.  

<p align="center">
<img src="docs/de_bd.png" alt="Disparity estimation process block diagram" width=50%>
</p>

## Implementation Details

* All stages of the algorithm are developed in both C++ and OpenCL.  
	* C++ parallelism is introduced via the POSIX threads (pthreads) library. Disparity level parallelism, enabling up to 64 concurrent threads, is supported.  
	* OpenCL parallelism is inherent through the concurrent execution of kernels on an OpenCL-compatible device. The optimum level of parallelism will be device-specific.  
* Support for live video disparity estimation using the OpenCV VideoCapture interface as well as standard static image computation.
* Embedded support for experimentation with the OpenCV standard Semi-Global Block Matching (SGBM) algorithm.

## Installation

### Prerequisites
* Hardware:
	* Development Platform - optionally but ideally including OpenCL compatible devices
* Software:
	* OpenCV 3.0.0 - [Installation in Linux instructions](http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html)
	* pthread and realtime libraries (lpthread, lrt)
	* [optional] OpenCL runtime and library

### Compilation 
* Download project folder and transfer to platform
* Enter the base directory using `cd DE_APP`
* Natively compile the project using `make -j8`. Adjust -j8 to suit the number of simultaneous threads supported on your platform.

### Deployment
* Run the application using `./bin/Release/DE_APP <program arguments>`
* The following program arguments must be specified:
	* Matching Algorithm type: 
		* STEREO_GIF - Guided Image Filter
		* STEREO_SGBM - Semi Global Block Matching
	* Media type:
		* VIDEO
		* IMAGE <left image filename> <right image filename>
* When specifying the VIDEO media type, the following optional arguments can be included:
	* RECALIBRATE - recalculate the intrinsic and extrinsic parameters of the stereo camera. Previously captured chessboard images must be supplied if the RECAPTURE flag is not also set.
	* RECAPTURE - record chessboard image pairs in preparation for calibration. A chessboard image must be presented in front of the stereo camera and in full view of both cameras.
* For example, to run with the guided image filter algorithm using a stereo camera, specify:
	* `./bin/Release/DE_APP STEREO_GIF VIDEO`
* To run with calibration and capture beforehand, specify:
	* `./bin/Release/DE_APP STEREO_GIF VIDEO RECALIBRATE RECAPTURE`
* Image disparity estimation is achieved using for example:
	* `./bin/Release/DE_APP STEREO_GIF IMAGE left_img.png right_img.png`
