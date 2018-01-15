/* This is sample from the OpenCV book. The copyright notice is below */

/* *************** License:**************************
   Oct. 3, 2008
   Right to use this code in any way you want without warranty, support or any guarantee of it working.

   BOOK: It would be nice if you cited it:
   Learning OpenCV: Computer Vision with the OpenCV Library
     by Gary Bradski and Adrian Kaehler
     Published by O'Reilly Media, October 3, 2008

   AVAILABLE AT:
     http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
     Or: http://oreilly.com/catalog/9780596516130/
     ISBN-10: 0596516134 or: ISBN-13: 978-0596516130

   OPENCV WEBSITES:
     Homepage:      http://opencv.org
     Online docs:   http://docs.opencv.org
     Q&A forum:     http://answers.opencv.org
     Issue tracker: http://code.opencv.org
     GitHub:        https://github.com/Itseez/opencv/
   ************************************************** */
 /*---------------------------------------------------------------------------
   StereoCalib.h - Stereo Camera Calibration Function Header
  ---------------------------------------------------------------------------
   Co-Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "ComFunc.h"

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <algorithm>
#include <iterator>
#include <ctype.h>

// Recalibrate
#define FILE_INTRINSICS 	BASE_DIR "data/intrinsics.yml"
#define FILE_EXTRINSICS 	BASE_DIR "data/extrinsics.yml"
#define FILE_CALIB_XML  	BASE_DIR "data/stereo_calib.xml"
// Recapture
#define FILE_TEMPLATE_LEFT	BASE_DIR "data/chessboard%dL.png"
#define FILE_TEMPLATE_RIGHT	BASE_DIR "data/chessboard%dR.png"

struct StereoCameraProperties{
    Mat cameraMatrix[2];
    Mat distCoeffs[2];
	Mat R, T, E, F; //rotation matrix, translation vector, essential matrix E=[T*R], fundamental matrix
	Mat R1, R2, P1, P2, Q;
	Size imgSize;
    Rect roi[2]; //region of interest
};

int print_help();

void StereoCalib(const vector<string>& imagelist, Size boardSize, StereoCameraProperties& props, bool useCalibrated, bool showRectified);

bool readStringList( const string& filename, vector<string>& l );

int calibrateCamera(int boardWidth, int boardHeight, StereoCameraProperties& props,	string imagelistfn);
