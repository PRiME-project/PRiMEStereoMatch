# PRiMEStereoMatch

## Theoretical Background

A heterogeneous and fully parallel stereo matching algorithm for depth estimation. Stereo Matching is based on the disparity estimation algorithm, an algorithm designed to calculate 3D depth information about a scene from a pair of 2D images captured by a stereoscopic camera. The algorithm contains the following stages:

* Cost Volume Construction - weighted absolute difference of colours and gradients function.
* Cost Volume Filtering - local adaptive support weight (ADSW) Guided Image Filter (GIF) function.  
* Disparity Selection - winner-takes-all minimum cost search and corresponding disparity selection.  
* Post Processing - left-right occlusion check, invalid pixel removal and weight-median filtering.  

![Disparity Esitmation Process](docs/de_bd.png "Disparity esitmation process block diagram" {width=40px height=400px})


## Implementation Details

* All stages of the algorithm are developed in both C++ and OpenCL.  
	* C++ parallelism is introduced via the POSIX threads (pthreads) library. Disparity level parallelism, enabling up to 64 concurrent threads, is supported.  
	* OpenCL parallelism is inherent through the concurrent execution of kernels on the OpenCL-compatible device. The optimum level of parallelism will be device-specific.  
* Support for live video disparity estimation using the OpenCV VideoCapture interface as well as standard static image computation.
* Embedded support for experimentation with the OpenCV standard Semi-Global Block Matching (SGBM) algorithm.
