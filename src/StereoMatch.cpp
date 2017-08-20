/*---------------------------------------------------------------------------
   StereoMatch.cpp - Stereo Matching Application Code/Class
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "StereoMatch.h"

//#############################################################################################################
//# SM Preprocessing that we don't want to repeat
//#############################################################################################################
StereoMatch::StereoMatch(int argc, char *argv[], int gotOpenCLDev)
{
    printf("Preprocessing for Stereo Matching.\n");
	//#############################################################################################################
    //# Setup - check input arguments
    //#############################################################################################################
    printf("Disparity Estimation for Depth Analysis in Stereo Vision Applications.\n");
	end_de = false;

	video = false;
	maxDis = 64;
	imgType = CV_32F;
	de_mode = OCV_DE;
	num_threads = MAX_CPU_THREADS;
	gotOCLDev = gotOpenCLDev;
	//MatchingAlgorithm = STEREO_SGBM;
	MatchingAlgorithm = STEREO_GIF;
	error_threshold = 4 * (256/maxDis);
	strcpy(left_img_filename, "../data/teddy2.png");
	strcpy(right_img_filename, "../data/teddy6.png");
	strcpy(gt_img_filename, "../data/teddy2_gt.png");

	//inputArgParser(argc, argv);

	if(video){
		cap = VideoCapture(0);
		if (cap.isOpened()){
			printf("Opened the VideoCapture device.\n");
			stereoCameraSetup();
		}
		else{
			printf("Could not open the VideoCapture device.\n");
			exit(1);
		}
	}
	else{
		//#############################################################################################################
		//# Image Loading
		//#############################################################################################################
        printf("Loading Images...\n");
		lFrame = imread(left_img_filename, CV_LOAD_IMAGE_COLOR);
		if(lFrame.empty())
		{
			printf("Failed to read left image \"%s\".\n", left_img_filename);
			exit(1);
		}
		else{
			printf("Loaded %s.\n", left_img_filename);
		}
		rFrame = imread(right_img_filename, CV_LOAD_IMAGE_COLOR);
		if(lFrame.empty())
		{
			printf("Failed to read right image \"%s\".\n", right_img_filename);
			exit(1);
		}
		else{
			printf("Loaded %s.\n", right_img_filename);
		}
		gtFrameImg = imread(gt_img_filename, CV_LOAD_IMAGE_COLOR);
		if(rFrame.empty()){
			printf("SM: Failed to read right image \"%s\".\n", gt_img_filename);
			exit(1);
		}else{
			printf("SM: Loaded %s.\n", gt_img_filename);
		}
		cvtColor(gtFrameImg, gtFrame, CV_RGB2GRAY);
		minMaxLoc(gtFrame, &minVal_gt, &maxVal_gt);
		gtFrame.convertTo(gtFrame, CV_8U, 255/(maxVal_gt - minVal_gt));
	}

	//#############################################################################################################
	//# Display Setup
	//#############################################################################################################
	//Set up display window to hold both input images and both output disparity maps
	resizeWindow("InputOutput", lFrame.cols*3, lFrame.rows*2); //Rectified image size - not camera resolution size
	display_container = Mat(lFrame.rows*2, lFrame.cols*3, CV_8UC3);
	leftInputImg  = Mat(display_container, Rect(0,             0,           lFrame.cols, lFrame.rows)); //Top Left
	rightInputImg = Mat(display_container, Rect(lFrame.cols,   0,           lFrame.cols, lFrame.rows)); //Top Right
	leftDispMap   = Mat(display_container, Rect(0,             lFrame.rows, lFrame.cols, lFrame.rows)); //Bottom Left
	rightDispMap  = Mat(display_container, Rect(lFrame.cols,   lFrame.rows, lFrame.cols, lFrame.rows)); //Bottom Right
	gtDispMap	  = Mat(display_container, Rect(lFrame.cols*2, 0, 			lFrame.cols, lFrame.rows)); //Top Far Right
	errDispMap	  = Mat(display_container, Rect(lFrame.cols*2, lFrame.rows,	lFrame.cols, lFrame.rows)); //Bottom Far Right

	lFrame.copyTo(leftInputImg);
	rFrame.copyTo(rightInputImg);

	if(!video){
		gtFrameImg.copyTo(gtDispMap);
		eDispMap = Mat(lFrame.rows, lFrame.cols, CV_8UC1);
	}
	imshow("InputOutput", display_container);

	//#############################################################################################################
    //# SGBM Mode Setup
    //#############################################################################################################
	setupOpenCVSGBM(lFrame.channels(), maxDis);
	imgDisparity16S = Mat(lFrame.rows, lFrame.cols, CV_16S);
	blankDispMap = Mat(rFrame.rows, rFrame.cols, CV_8UC3);

	//#############################################################################################################
    //# GIF Mode Setup
    //#############################################################################################################
	if(imgType == CV_32F){
		// Frame Preprocessing
		cvtColor( lFrame, lFrame, CV_BGR2RGB );
		cvtColor( rFrame, rFrame, CV_BGR2RGB );

		lFrame.convertTo( lFrame, CV_32F, 1 / 255.0f );
		rFrame.convertTo( rFrame, CV_32F,  1 / 255.0f );
	}
	SMDE = new DispEst(lFrame, rFrame, maxDis, num_threads, gotOCLDev);

	//#############################################################################################################
	//# End of Preprocessing (that we don't want to repeat)
    //#############################################################################################################
	//printf("End of Preprocessing\n");

    printf("SM: StereoMatch Application Initialised\n");
	return;
}

StereoMatch::~StereoMatch(void)
{
	printf("SM: Shutting down StereoMatch Application\n");
	//delete ssgbm;
	delete SMDE;
	if(video) cap.release();
	printf("SM: Application Shut down\n");
}

float get_rt(){
	struct timespec realtime;
	clock_gettime(CLOCK_MONOTONIC,&realtime);
	return (float)(realtime.tv_sec*1000000+realtime.tv_nsec/1000);
}

int StereoMatch::updateFrameType(void)
{
	int prev_type = lFrame.type() & CV_MAT_DEPTH_MASK;
	// frames need converting every time for video
	// or when the STEREO_GIF imgType changes
	if(video || (prev_type != imgType))
	{
		if(imgType == CV_32F)
		{
			cvtColor( lFrame, lFrame, CV_BGR2RGB );
			cvtColor( rFrame, rFrame, CV_BGR2RGB );
			lFrame.convertTo( lFrame, CV_32F, 1 / 255.0f );
			rFrame.convertTo( rFrame, CV_32F,  1 / 255.0f );
		}
		else if(imgType == CV_8U)
		{
			lFrame.convertTo( lFrame, CV_8U, 255);
			rFrame.convertTo( rFrame, CV_8U, 255);
		}
	}

	if(prev_type == imgType)	return 0; //return if type is the same as last frame

	delete SMDE;
    printf("SM: Re-constructing SMDE Object.\n");
	SMDE = new DispEst(lFrame, rFrame, maxDis, num_threads, gotOCLDev);
	return 0;
}

//#############################################################################################################
//# Complete GIF stereo matching process
//#############################################################################################################
int StereoMatch::Compute()
{
	//printf("Computing Depth Map\n");

	if(MatchingAlgorithm == STEREO_GIF)
		updateFrameType();

	de_time = get_rt();
	//#############################################################################################################
	//# Frame Capture and Preprocessing
	//#############################################################################################################
	if(video)
	{
		cap >> vFrame; //capture a frame from the camera
		if(!vFrame.data)
		{
			printf("Could not load video frame\n");
			return -1;
		}

		lFrame = vFrame(Rect(0,0, vFrame.cols/2,vFrame.rows)); //split the frame into left
		rFrame = vFrame(Rect(vFrame.cols/2, 0, vFrame.cols/2, vFrame.rows)); //and right images

		if(!lFrame.data || !rFrame.data)
		{
			printf("No data in left or right frames\n");
			return -1;
		}

		//Applies a generic geometrical transformation to an image.
		//http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#remap
		remap(lFrame, lFrame_rec, mapl[0], mapl[1], INTER_LINEAR);
		remap(rFrame, rFrame_rec, mapr[0], mapr[1], INTER_LINEAR);

		lFrame = lFrame_rec(cropBox);
		rFrame = rFrame_rec(cropBox);

		lFrame.copyTo(leftInputImg);
		rFrame.copyTo(rightInputImg);
	}

	//#############################################################################################################
	//# Start of Disparity Map Creation
	//#############################################################################################################
	if(MatchingAlgorithm == STEREO_SGBM)
	{
		if((lFrame.type() & CV_MAT_DEPTH_MASK) != CV_8U){
			lFrame.convertTo(lFrame, CV_8U, 255);
			rFrame.convertTo(rFrame, CV_8U, 255);
		}

		//OpenCV code
		ssgbm->compute(lFrame, rFrame, imgDisparity16S); //Compute the disparity map
		minMaxLoc(imgDisparity16S, &minVal, &maxVal); //Check its extreme values
		imgDisparity16S.convertTo(lDispMap, CV_8U, 255/(maxVal - minVal));
		//lDispMap.copyTo(leftDispMap); //Load the disparity map to the display
		//applyColorMap( leftDispMap, leftDispMap, COLORMAP_JET);
		cvtColor(lDispMap, leftDispMap, CV_GRAY2RGB);
	}
	else if(MatchingAlgorithm == STEREO_GIF)
	{
		updateFrameType();

		SMDE->lImg = lFrame;
		SMDE->rImg = rFrame;
		SMDE->threads = num_threads;

		// ******** Disparity Estimation Code ******** //
		//printf("Disparity Estimation Started..\n");

		if(de_mode == OCV_DE || !gotOCLDev)
		{
			cvc_time = get_rt();
			SMDE->CostConst_CPU();
			cvc_time = get_rt() - cvc_time;

			cvf_time = get_rt();
			SMDE->CostFilter_CPU();
			cvf_time = get_rt()- cvf_time;

			dispsel_time = get_rt();
			SMDE->DispSelect_CPU();
			dispsel_time = get_rt() - dispsel_time;

			pp_time = get_rt();
			SMDE->PostProcess_CPU();
			pp_time = get_rt() - pp_time;
		}
		else
		{
			cvc_time = get_rt();
			SMDE->CostConst_GPU();
			cvc_time = get_rt() - cvc_time;

			cvf_time = get_rt();
			SMDE->CostFilter_GPU();
			cvf_time = get_rt()- cvf_time;

			dispsel_time = get_rt();
			SMDE->DispSelect_GPU();
			dispsel_time = get_rt() - dispsel_time;

			pp_time = get_rt();
			SMDE->PostProcess_GPU();
			pp_time = get_rt() - pp_time;
		}

		// ******** Show Disparity Map  ******** //
		//applyColorMap( SMDE->lDisMap*4, lDispMap, COLORMAP_JET); // *4 for conversion from disparty range (0-64) to RGB char range (0-255)
		//lDispMap.copyTo(leftDispMap); //copy to leftDispMap display rectangle
		minMaxLoc(SMDE->lDisMap, &minVal, &maxVal);
		SMDE->lDisMap.convertTo(lDispMap, CV_8U, 4);//255/(maxVal - minVal));
		cvtColor(lDispMap, leftDispMap, CV_GRAY2RGB);

		//applyColorMap( SMDE->rDisMap*4, rDispMap, COLORMAP_JET);
		//rDispMap.copyTo(rightDispMap); //copy to rightDispMap display rectangle
		SMDE->rDisMap.convertTo(rDispMap, CV_8U, 255/(maxVal - minVal));
		cvtColor(rDispMap, rightDispMap, CV_GRAY2RGB);
		// ******** Show Disparity Map  ******** //

//		printf("CVC Time: %.2f ms\n",cvc_time/1000);
//		printf("CVF Time: %.2f ms\n",cvf_time/1000);
//		printf("DispSel Time: %.2f ms\n",dispsel_time/1000);
//		printf("PP Time: %.2f ms\n",pp_time/1000);

//		imwrite("leftDisparityMap.png", leftDispMap);
//		imwrite("rightDisparityMap.png", rightDispMap);

	}
	printf("DE Time: %.2f ms\n",(get_rt() - de_time)/1000);
	//Perform these steps for all algorithms:
	imshow("InputOutput", display_container);

	if(!video){
		//Check pixel errors against ground truth depth map here.
		//Can only be done with images as golden reference is required.
		float num_bad_pixels = 0;
		float avg_err = 0;

		for(int y = 0; y < gtFrame.rows; y++)
		{
			uchar* eData = (uchar*) eDispMap.ptr<uchar>( y );
			uchar* lData = (uchar*) lDispMap.ptr<uchar>( y );
			uchar* gtData = (uchar*) gtFrame.ptr<uchar>( y );
			for(int x = 0; x < gtFrame.cols; x++)
			{
				//printf("lDispMap[%d][%d] = %d\n", y, x, lData[x]);
				//printf("gtFrame[%d][%d] = %d\n", y, x, gtData[x]);
				eData[x] = abs(lData[x] - gtData[x]);
				avg_err += (float)eData[x];
				if(eData[x] > error_threshold){
					num_bad_pixels++;
				}
				else{
					eData[x] = 0;
				}
			}
			//exit(1);
		}
		float num_pixels = gtFrame.cols*gtFrame.rows;
		printf("percent_bad_pixels = %.2f\%\n", (float)num_bad_pixels*100/num_pixels);
		printf("avg err = %.2f\n", (float)avg_err/num_pixels);

		minMaxLoc(eDispMap, &minVal, &maxVal);
		//eDispMap.convertTo(eDispMap, CV_8U, 255/(maxVal - minVal));
		cvtColor(eDispMap, errDispMap, CV_GRAY2RGB);
		imshow("InputOutput", display_container);
	}
	return de_time;
}

//#############################################################################################################
//# Calibration and Paramter loading for stereo camera setup
//#############################################################################################################
int StereoMatch::stereoCameraSetup(void)
{
	cap.set(CAP_PROP_FRAME_HEIGHT, 480); //480, 720, 1080, 1242
	cap.set(CAP_PROP_FRAME_WIDTH, 1280); //1280, 2560, 3840, 4416
	cout << "CAP_PROP_FRAME_HEIGHT: " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
	cout << "CAP_PROP_FRAME_WIDTH: " << cap.get(CAP_PROP_FRAME_WIDTH) << endl;

	cap >> vFrame;
	if(!vFrame.data)
	{
		printf("Could not load video frame\n");
		exit(1);
	}

	lFrame = vFrame(Rect(0,0, vFrame.cols/2,vFrame.rows));
	rFrame = vFrame(Rect(vFrame.cols/2, 0, vFrame.cols/2, vFrame.rows));
	if(!lFrame.data || !rFrame.data)
	{
		printf("No data in left or right frames\n");
		exit(1);
	}

	if(recalibrate)
    {
		resizeWindow("InputOutput", lFrame.cols*2, lFrame.rows*2); //Native camera image resolution
		display_container = Mat(lFrame.rows*2, lFrame.cols*2, CV_8UC3);
		leftInputImg  = Mat(display_container, Rect(0,           0,           lFrame.cols, lFrame.rows)); //Top Left
		rightInputImg = Mat(display_container, Rect(lFrame.cols, 0,           lFrame.cols, lFrame.rows)); //Top Right
		leftDispMap   = Mat(display_container, Rect(0,           lFrame.rows, lFrame.cols, lFrame.rows)); //Bottom Left
		rightDispMap  = Mat(display_container, Rect(lFrame.cols, lFrame.rows, lFrame.cols, lFrame.rows)); //Bottom Right

		lFrame.copyTo(leftInputImg);
		rFrame.copyTo(rightInputImg);
		imshow("InputOutput", display_container);

		//#############################################################################################################
		//# Camera Calibration - find intrinsic & extrinsic parameters from first principles (contains rectification)
		//#############################################################################################################
		if(recaptureChessboards)
		{
			//Capture a series of calibration images from the camera.
			printf("Capturing chessboard images for calibration.\n");
			printf("A chessboard image with 9x6 inner corners should be placed in the view of the camera.\n");
			captureChessboards();
		}
		printf("Running Calibration.\n");
		calibrateCamera(9, 6, camProps, FILE_CALIB_XML);
		printf("Calibration Complete.\n");
	}
	else
	{
		//#############################################################################################################
		//# Camera Setup - load existing intrinsic & extrinsic parameters
		//#############################################################################################################
		string intrinsic_filename = FILE_INTRINSICS;
		string extrinsic_filename = FILE_EXTRINSICS;

		// Read in intrinsic parameters
		printf("Loading intrinsic parameters.\n");
        FileStorage fs(intrinsic_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", intrinsic_filename.c_str());
            return -1;
        }
        fs["M1"] >> camProps.cameraMatrix[0];
        fs["D1"] >> camProps.distCoeffs[0];
        fs["M2"] >> camProps.cameraMatrix[1];
        fs["D2"] >> camProps.distCoeffs[1];

		// Read in extrinsic parameters
		printf("Loading extrinsic parameters.\n");
        fs.open(extrinsic_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", extrinsic_filename.c_str());
            return -1;
        }

        fs["R"] >> camProps.R;

        fs["T"] >> camProps.T;

		printf("Performing Stereo Rectify.\n");
		camProps.imgSize = lFrame.size();
		//stereoRectify performed inside calibration function atm but not necessary in finding extrinsics that can be loaded
		stereoRectify(camProps.cameraMatrix[0], camProps.distCoeffs[0], camProps.cameraMatrix[1], camProps.distCoeffs[1],
						camProps.imgSize, camProps.R, camProps.T, camProps.R1, camProps.R2, camProps.P1, camProps.P2, camProps.Q,
						CALIB_ZERO_DISPARITY, 1, camProps.imgSize, &camProps.roi[0], &camProps.roi[1]);
	}

	///Computes the undistortion and rectification transformation map.
	///http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#initundistortrectifymap
	//Left
	initUndistortRectifyMap(camProps.cameraMatrix[0], camProps.distCoeffs[0], camProps.R1, camProps.P1, camProps.imgSize, CV_16SC2, mapl[0], mapl[1]);
	//Right
	initUndistortRectifyMap(camProps.cameraMatrix[1], camProps.distCoeffs[1], camProps.R2, camProps.P2, camProps.imgSize, CV_16SC2, mapr[0], mapr[1]);

	///Applies a generic geometrical transformation to an image.
	///http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#remap
	remap(lFrame, lFrame_rec, mapl[0], mapl[1], INTER_LINEAR);
	remap(rFrame, rFrame_rec, mapr[0], mapr[1], INTER_LINEAR);

	//Use create ROI which is valid in both left and right ROIs
	int tl_x = MAX(camProps.roi[0].x, camProps.roi[1].x);
	int tl_y = MAX(camProps.roi[0].y, camProps.roi[1].y);
	int br_x = MIN(camProps.roi[0].width + camProps.roi[0].x, camProps.roi[1].width + camProps.roi[1].x);
	int br_y = MIN(camProps.roi[0].height + camProps.roi[0].y, camProps.roi[1].height + camProps.roi[1].y);
	cropBox = Rect(Point(tl_x, tl_y), Point(br_x, br_y));

	lFrame = lFrame_rec(cropBox);
	rFrame = rFrame_rec(cropBox);

	return 0;
}

//#############################################################################################################
//# Chessboard image capture for camera calibration
//#############################################################################################################
int StereoMatch::captureChessboards(void)
{
    int img_num = 0;
	char imageLoc[100];

	while(img_num < 10)
	{
		cap >> vFrame;

		if(!vFrame.data)
		{
			printf("Cannot capture data from video source\n");
			img_num = 10;
			return -1;
		}

		lFrame = vFrame(Rect(0,0, vFrame.cols/2,vFrame.rows));
		rFrame = vFrame(Rect(vFrame.cols/2, 0, vFrame.cols/2, vFrame.rows));
		lFrame.copyTo(leftInputImg);
		rFrame.copyTo(rightInputImg);

        if(cap_key == 'r')
        {
			sprintf(imageLoc, FILE_TEMPLATE_LEFT, img_num);
			imwrite(imageLoc, lFrame);
			lFrame.copyTo(leftDispMap);

			sprintf(imageLoc, FILE_TEMPLATE_RIGHT, img_num++);
			imwrite(imageLoc, rFrame);
			rFrame.copyTo(rightDispMap);
        }
        imshow("InputOutput", display_container);
        cap_key = waitKey(5);
        if(cap_key == 'q')
			exit(1);
	}
	return 0;
}

int StereoMatch::inputArgParser(int argc, char *argv[])
{
	if( argc < 2 ) {
        printf("\nInput Argument Error: Please specify the Media Type as a minimum requirement:\n" );
        printf("Usage: ./DE_APP VIDEO ( [RECALIBRATE?] [RECAPTURE] )\n" );
        printf("Usage: \t or");
        printf("Usage: ./DE_APP IMAGE left_image_filename right_image_filename\n" );
		exit(1);
	}
	printf("Matching Algorithm Type: %s.\n", MatchingAlgorithm ? "STEREO_GIF" : "STEREO_SGBM");

	if(!strcmp(argv[1],"VIDEO")){
		printf("Media Type: \t\t Video Processing from Stereo Camera.\n");
		video = true;
		recalibrate = false;
		if(argc > 2){
			if(!strcmp(argv[2],"RECALIBRATE")) {
				printf("Recalibrating...\n");
				recalibrate = true;
			}
		}
		recaptureChessboards = false;
		if(argc > 3){
			if(!strcmp(argv[3],"RECAPTURE")) {
				printf("Recapturing...\n");
				recaptureChessboards = true;
			}
		}
	}
	else if(!strcmp(argv[1],"IMAGE"))
	{
		printf("Media Type: \t\t Image Processing from Static Image.\n");
		if( argc < 3 )
		{
			printf("Please specify the image filenames to use, e.g. left_img.png right_img.png\n");
			printf("Usage: ./DE_APP IMAGE left_image_filename right_image_filename\n" );
			exit(1);
		}
		else
		{
			if((strlen(argv[2]) > 100) || (strlen(argv[3]) > 100))
			{
				printf("Left or right image filename is too long, please shorten to fewer than 100 char and retry.\n");
				exit(1);

			}
			if((strlen(argv[2]) < 4) || (strlen(argv[3]) < 4))
			{
				printf("Left or right image filename is too short, did you forget to include the file extension?\n");
				exit(1);

			}
			char *img_ext = &argv[2][strlen(argv[2])-4];
			if(!strcmp(img_ext, ".png") || !strcmp(img_ext, ".jpg") || !strcmp(img_ext, ".ppm"))
			{
				strcpy(left_img_filename, argv[2]);
				printf("Left Image : %s\n", left_img_filename);
			}
			else
			{
				printf("Left Image: Incompatible image filename extension specified. \nPlease use either .png, .ppm .jpg images\n");
				printf("Usage: ./DE_APP IMAGE left_image_filename right_image_filename\n" );
				exit(1);
			}

			img_ext = &argv[3][strlen(argv[3])-4];
			if(!strcmp(img_ext, ".png") || !strcmp(img_ext, ".jpg") || !strcmp(img_ext, ".ppm"))
			{
				strcpy(right_img_filename, argv[3]);
				printf("Right Image : %s\n", right_img_filename);
			}
			else
			{
				printf("Right Image: Incompatible image filename extension specified. \nPlease use either .png or jpg images\n");
				printf("Usage: ./DE_APP IMAGE left_image_filename right_image_filename\n" );
				exit(1);
			}
			video = false;
		}
	}
	else{
		printf("Invalid media type chosen:\n");
		printf("Usage: ./DE_APP [MEDIA TYPE = VIDEO|IMAGE [image_filenames]] ([RECALIBRATE?] [RECAPTURE])\n" );
		exit(1);
	}
	return 0;
}

//#############################################################################################################
//# Setup for OpenCV implementation of Stereo matching using Semi-Global Block Matching (SGBM)
//#############################################################################################################
int StereoMatch::setupOpenCVSGBM(int channels, int ndisparities)
{
	int mindisparity = 0;
	int SADWindowSize = 5;

	// Call the constructor for StereoSGBM
	ssgbm = StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);

    ssgbm->setP1(8*channels*SADWindowSize*SADWindowSize);
    ssgbm->setP2(32*channels*SADWindowSize*SADWindowSize);
    ssgbm->setDisp12MaxDiff(1);
    ssgbm->setPreFilterCap(63);
	ssgbm->setUniquenessRatio(10);
    ssgbm->setSpeckleWindowSize(100);
    ssgbm->setSpeckleRange(32);
	//ssgbm->setMode(StereoSGBM::MODE_HH); // enum{ MODE_SGBM = 0, MODE_HH = 1, MODE_SGBM_3WAY = 2 }
	ssgbm->setMode(StereoSGBM::MODE_HH);
    return 0;
}
