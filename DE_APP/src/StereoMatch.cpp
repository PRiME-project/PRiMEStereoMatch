/*---------------------------------------------------------------------------
   StereoMatch.cpp - Stereo Matching Application Code/Class
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
  ---------------------------------------------------------------------------*/
#include "StereoMatch.h"

StereoMatch::StereoMatch(int argc, char *argv[], bool gotOpenCLDev)
{
	//#############################################################################################################
    //# Setup - check input arguments
    //#############################################################################################################
    printf("Disparity Estimation for Depth Analysis in Stereo Vision Applications.\n");
	end_de = false;
	maxDis = 64;
	de_mode = OCV_DE;
	num_threads = MAX_CPU_THREADS;
	filter = true;
	gotOCLDev = gotOpenCLDev;

	inputArgParser(argc, argv);

	//#############################################################################################################
	//# Preprocessing that we don't want to repeat
    //#############################################################################################################
    printf("Preprocessing for Stereo Matching.\n");

	if(video)
	{
		stereoCameraSetup();
	}
	else
	{
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
	}

	//#############################################################################################################
	//# Display Setup
	//#############################################################################################################
	//Set up display window to hold both input images and both output disparity maps
	resizeWindow("InputOutput", lFrame.cols*2, lFrame.rows*2); //Rectified image size - not camera resolution size
	display_container = Mat(lFrame.rows*2, lFrame.cols*2, CV_8UC3);
	leftInputImg  = Mat(display_container, Rect(0,           0,           lFrame.cols, lFrame.rows)); //Top Left
	rightInputImg = Mat(display_container, Rect(lFrame.cols, 0,           lFrame.cols, lFrame.rows)); //Top Right
	leftDispMap   = Mat(display_container, Rect(0,           lFrame.rows, lFrame.cols, lFrame.rows)); //Bottom Left
	rightDispMap  = Mat(display_container, Rect(lFrame.cols, lFrame.rows, lFrame.cols, lFrame.rows)); //Bottom Right

	lFrame.copyTo(leftInputImg);
	rFrame.copyTo(rightInputImg);

	imshow("InputOutput", display_container);


	//#############################################################################################################
    //# SGBM Mode Setup
    //#############################################################################################################
	if(MatchingAlgorithm == STEREO_SGBM)
	{
		setupOpenCVSGBM(lFrame.channels(), maxDis*2);
		imgDisparity16S = Mat( lFrame.rows, lFrame.cols, CV_16S );
	}
	//#############################################################################################################
    //# GIF Mode Setup
    //#############################################################################################################
	else if(MatchingAlgorithm == STEREO_GIF)
	{
		// Frame Preprocessing
		cvtColor( lFrame, lFrame, CV_BGR2RGB );
		cvtColor( rFrame, rFrame, CV_BGR2RGB );

		lFrame.convertTo( lFrame, CV_32F, 1 / 255.0f );
		rFrame.convertTo( rFrame, CV_32F,  1 / 255.0f );

		SMDE = new DispEst(lFrame, rFrame, maxDis, num_threads, gotOCLDev);
	}
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
	if(MatchingAlgorithm == STEREO_SGBM) delete ssgbm;
	if(MatchingAlgorithm == STEREO_GIF) delete SMDE;
	cap.release();
	printf("SM: Application Shut down\n");
}

//#############################################################################################################
//# Calibration and Paramter loading for stereo camera setup
//#############################################################################################################
int StereoMatch::stereoCameraSetup(void)
{
	cap = VideoCapture(0);
	if (!cap.isOpened())
	{
		printf("Cannot open the VideoCapture object\n");
		exit(1);
	}
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
			printf("A chessboard image with 7x7 inner corners should be placed in the view of the camera.\n");
			captureChessboards();
		}
		printf("Running Calibration.\n");
		calibrateCamera(7, 7, camProps, "data/stereo_calib.xml");
		printf("Calibration Complete.\n");
	}
	else
	{
		//#############################################################################################################
		//# Camera Setup - load existing intrinsic & extrinsic parameters
		//#############################################################################################################
		string intrinsic_filename = "data/intrinsics.yml";
		string extrinsic_filename = "data/extrinsics.yml";

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

	cropBox = Rect(MAX(camProps.roi[0].x, camProps.roi[1].x), MAX(camProps.roi[0].y, camProps.roi[1].y),
	MIN(camProps.roi[0].width, camProps.roi[1].width), MIN(camProps.roi[0].height, camProps.roi[1].height));

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
			sprintf(imageLoc, "data/chessboard%dL.png", img_num);
			imwrite(imageLoc, lFrame);
			lFrame.copyTo(leftDispMap);

			sprintf(imageLoc, "data/chessboard%dR.png", img_num++);
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

//#############################################################################################################
//# Setup for OpenCV implementation of Stereo matching using Semi-Global Block Matching (SGBM)
//#############################################################################################################
int StereoMatch::setupOpenCVSGBM(int channels, int ndisparities)
{
	int mindisparity = 0;
	int SADWindowSize = 9;

	// Call the constructor for StereoSGBM
	ssgbm = StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);

    ssgbm->setP1(8*channels*SADWindowSize*SADWindowSize);
    ssgbm->setP2(32*channels*SADWindowSize*SADWindowSize);
    ssgbm->setDisp12MaxDiff(1);
    ssgbm->setPreFilterCap(63);
	ssgbm->setUniquenessRatio(10);
    ssgbm->setSpeckleWindowSize(100);
    ssgbm->setSpeckleRange(32);
	ssgbm->setMode(StereoSGBM::MODE_HH);
	//ssgbm->setMode(StereoSGBM::MODE_SGBM);

    return 0;
}

float get_rt(){
	struct timespec realtime;
	clock_gettime(CLOCK_MONOTONIC,&realtime);
	return (float)(realtime.tv_sec*1000000+realtime.tv_nsec/1000);
}

//#############################################################################################################
//# Complete GIF stereo matching process
//#############################################################################################################
int StereoMatch::Compute()
{
	printf("Computing Depth Map\n");
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

		///Applies a generic geometrical transformation to an image.
		///http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#remap
		remap(lFrame, lFrame_rec, mapl[0], mapl[1], INTER_LINEAR);
		remap(rFrame, rFrame_rec, mapr[0], mapr[1], INTER_LINEAR);

		lFrame = lFrame_rec(cropBox);
		rFrame = rFrame_rec(cropBox);

		lFrame.copyTo(leftInputImg);
		rFrame.copyTo(rightInputImg);

		if(MatchingAlgorithm == STEREO_GIF)
		{
			cvtColor( lFrame, lFrame, CV_BGR2RGB );
			cvtColor( rFrame, rFrame, CV_BGR2RGB );

			lFrame.convertTo( lFrame, CV_32F, 1 / 255.0f );
			rFrame.convertTo( rFrame, CV_32F,  1 / 255.0f );
		}
	}

	//#############################################################################################################
	//# Start of Disparity Map Creation
	//#############################################################################################################
	if(MatchingAlgorithm == STEREO_SGBM)
	{
		///OpenCV code
		ssgbm->compute( lFrame, rFrame, imgDisparity16S );
		//-- Check its extreme values
		minMaxLoc( imgDisparity16S, &minVal, &maxVal );
		imgDisparity16S.convertTo(lDispMap, CV_8UC1, 255/(maxVal - minVal));
		cvtColor( lDispMap, lDispMap, CV_GRAY2BGR);
		lDispMap.copyTo(leftDispMap);
	}
	else if(MatchingAlgorithm == STEREO_GIF)
	{
		SMDE->lImg = lFrame;
		SMDE->rImg = rFrame;
		SMDE->threads = num_threads;

		// ******** Cost Volume Construction Code ******** //
		printf("Cost Volume Construction Started..\n");
		clock_gettime(CLOCK_MONOTONIC,&realtime);
		cvc_start=realtime.tv_sec*1000000+realtime.tv_nsec/1000;

		if(de_mode == OCV_DE || !gotOCLDev)
			SMDE->CostConst_CPU();
		else
			SMDE->CostConst_GPU();

		clock_gettime(CLOCK_MONOTONIC,&realtime);
		cvc_end=realtime.tv_sec*1000000+realtime.tv_nsec/1000;
		cvc_time = cvc_end - cvc_start;
		fprintf(stderr, "Cost Volume Construction Done!\n");

		// ******** Cost Volume Filtering Code ******** //
		if(filter){
			fprintf(stderr, "Cost Volume Filtering Started..\n");
			clock_gettime(CLOCK_MONOTONIC,&realtime);
			cvf_start=realtime.tv_sec*1000000+realtime.tv_nsec/1000;

			if(de_mode == OCV_DE || !gotOCLDev)
				SMDE->CostFilter_CPU();
			else
				SMDE->CostFilter_GPU();

			clock_gettime(CLOCK_MONOTONIC,&realtime);
			cvf_end=realtime.tv_sec*1000000+realtime.tv_nsec/1000;
			cvf_time = cvf_end - cvf_start;
			fprintf(stderr, "Cost Volume Filtering Done!\n");
		}
		// ******** Disparity Selection Code ******** //
		fprintf(stderr, "Disparity Selection Started...\n");
		clock_gettime(CLOCK_MONOTONIC,&realtime);
		dispsel_start=realtime.tv_sec*1000000+realtime.tv_nsec/1000;

		if(de_mode == OCV_DE || !gotOCLDev)
			SMDE->DispSelect_CPU();
		else
			SMDE->DispSelect_GPU();

		clock_gettime(CLOCK_MONOTONIC,&realtime);
		dispsel_end=realtime.tv_sec*1000000+realtime.tv_nsec/1000;
		dispsel_time = dispsel_end - dispsel_start;
		fprintf(stderr, "Disparity Selection Done!\n");

		// ******** Post Processing Code ******** //
//			fprintf(stderr, "Post Processing Disparity Map...\n");
		clock_gettime(CLOCK_MONOTONIC,&realtime);
		pp_start=realtime.tv_sec*1000000+realtime.tv_nsec/1000;

		//SMDE->PostProcess();

		clock_gettime(CLOCK_MONOTONIC,&realtime);
		pp_end=realtime.tv_sec*1000000+realtime.tv_nsec/1000;
//			fprintf(stderr, "Post Processing Done!\n");
		pp_time = pp_end - pp_start;

		// ******** Show Disparity Map  ******** //
		lDispMap = SMDE->getLDisMap();
		cvtColor( lDispMap, lDispMap, CV_GRAY2BGR);
		lDispMap.copyTo(leftDispMap);
		rDispMap = SMDE->getRDisMap();
		cvtColor( rDispMap, rDispMap, CV_GRAY2BGR);
		rDispMap.copyTo(rightDispMap);
		// ******** Show Disparity Map  ******** //

		printf("CVC Time: %.2f ms\n",cvc_time/1000);
		printf("CVF Time: %.2f ms\n",cvf_time/1000);
		printf("DispSel Time: %.2f ms\n",dispsel_time/1000);
		printf("PP Time: %.2f ms\n",pp_time/1000);
		printf("DE Time: %.2f ms\n",(get_rt() - de_time)/1000);

	}
	//Perform these steps for all algorithms:
	imshow("InputOutput", display_container);

	return de_time;
}

void StereoMatch::inputArgParser(int argc, char *argv[])
{
	if( argc < 3 ) {
        printf("\nPlease specify a Matching Algorithm and Media Type as a minimum requirement:\n" );
        printf("Usage: ./<prog_name> [Matching Algorithm = STEREO_SGBM|STEREO_GIF] VIDEO ( [RECALIBRATE?] [RECAPTURE] )\n" );
        printf("Usage: \t or");
        printf("Usage: ./<prog_name> [Matching Algorithm = STEREO_SGBM|STEREO_GIF] IMAGE left_image_filename right_image_filename\n" );
		exit(1);
	}

	if(!strcmp(argv[1],"STEREO_SGBM")){
		printf("Matching Algorithm: \t STEREO_SGBM.\n");
		MatchingAlgorithm = STEREO_SGBM;
	}
	else if(!strcmp(argv[1],"STEREO_GIF")){
		printf("Matching Algorithm: \t STEREO_GIF.\n");
		MatchingAlgorithm = STEREO_GIF;
	}
	else{
		printf("Invalid matching algorithm chosen:\n");
		printf("Usage: ./<prog_name> [Matching Algorithm = STEREO_SGBM|STEREO_GIF] [MEDIA TYPE]\n" );
		exit(1);
	}
	if(!strcmp(argv[2],"VIDEO")){
		printf("Media Type: \t\t Video Processing from Stereo Camera.\n");
		video = true;

		if(argc > 3){
			if(!strcmp(argv[3],"RECALIBRATE")) {
				printf("Recalibrating...\n");
				recalibrate = true;
			}
		}
		if(argc > 4){
			if(!strcmp(argv[4],"RECAPTURE")) {
				printf("Recapturing...\n");
				recaptureChessboards = true;
			}
		}
	}
	else if(!strcmp(argv[2],"IMAGE"))
	{
		printf("Media Type: \t\t Image Processing from Static Image.\n");
		if( argc < 4 )
		{
			printf("Please specify the image filenames to use, e.g. left_img.png right_img.png\n");
			printf("Usage: ./<prog_name> [Matching Algorithm = STEREO_SGBM|STEREO_GIF] IMAGE left_image_filename right_image_filename\n" );
			exit(1);
		}
		else
		{
			if((strlen(argv[3]) > 100) || (strlen(argv[4]) > 100))
			{
				printf("Left or right image filename is too long, please shorten to fewer than 100 char and retry.\n");
				exit(1);

			}
			if((strlen(argv[3]) < 4) || (strlen(argv[4]) < 4))
			{
				printf("Left or right image filename is too short, did you forget to include the file extension?\n");
				exit(1);

			}
			char *img_ext = &argv[3][strlen(argv[3])-4];
			if(!strcmp(img_ext, ".png") || !strcmp(img_ext, ".jpg"))
			{
				strcpy(left_img_filename, argv[3]);
				printf("Left Image : %s\n", left_img_filename);
			}
			else
			{
				printf("Left Image: Incompatible image filename extension specified. \nPlease use either .png or .jpg images\n");
				printf("Usage: ./<prog_name> [Matching Algorithm = STEREO_SGBM|STEREO_GIF] IMAGE left_image_filename right_image_filename\n" );
				exit(1);
			}

			img_ext = &argv[4][strlen(argv[4])-4];
			if(!strcmp(img_ext, ".png") || !strcmp(img_ext, ".jpg"))
			{
				strcpy(right_img_filename, argv[4]);
				printf("Right Image : %s\n", right_img_filename);
			}
			else
			{
				printf("Right Image: Incompatible image filename extension specified. \nPlease use either .png or jpg images\n");
				printf("Usage: ./<prog_name> [Matching Algorithm = STEREO_SGBM|STEREO_GIF] IMAGE left_image_filename right_image_filename\n" );
				exit(1);
			}
			video = false;
		}
	}
	else{
		printf("Invalid media type chosen:\n");
		printf("Usage: ./<prog_name> [Matching Algorithm] [MEDIA TYPE = VIDEO|IMAGE [image_filenames]] ([RECALIBRATE?] [RECAPTURE])\n" );
		exit(1);
	}
}
