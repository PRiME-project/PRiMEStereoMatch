/*---------------------------------------------------------------------------
   StereoMatch.cpp - Stereo Matching Application Code/Class
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "StereoMatch.h"


unsigned long long get_timestamp()
{
	auto now = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
	return duration.count();
}

//#############################################################################
//# SM Preprocessing that we don't want to repeat
//#############################################################################
StereoMatch::StereoMatch(int argc, char *argv[], int gotOpenCLDev) :
	end_de(false)
{
    printf("Disparity Estimation for Depth Analysis in Stereo Vision Applications.\n");
    printf("Preprocessing for Stereo Matching.\n");

	//#########################################################################
    //# Setup - check input arguments
    //#########################################################################
	maxDis = 64;
	de_mode = OCL_DE;
	//de_mode = OCV_DE;
	media_mode = DE_IMAGE;
	//num_threads = MIN_CPU_THREADS;
	num_threads = MAX_CPU_THREADS;
	gotOCLDev = gotOpenCLDev;
	MatchingAlgorithm = STEREO_SGBM;
	//MatchingAlgorithm = STEREO_GIF;
	error_threshold = 4 * (256/maxDis);

	left_img_filename = string(BASE_DIR) + string("data/tsukuba3.ppm");
	right_img_filename = string(BASE_DIR) + string("data/tsukuba5.ppm");
	gt_img_filename = string(BASE_DIR) + string("data/tsukuba3_gt.pgm");

	left_img_filename = string(BASE_DIR) + string("data/teddy2.png");
	right_img_filename = string(BASE_DIR) + string("data/teddy6.png");
	gt_img_filename = string(BASE_DIR) + string("data/teddy2_gt.png");

	//inputArgParser(argc, argv);

	if(media_mode == DE_VIDEO){
		//#####################################################################
		//# Video Loading
		//#####################################################################
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
		//#####################################################################
		//# Image Loading
		//#####################################################################
        std::cout << "Loading Images...\n" << std::endl;
		lFrame = imread(left_img_filename).getUMat(ACCESS_READ);
		if(lFrame.empty())
		{
			std::cout << "Failed to read left image \"" << left_img_filename << "\"" << std::endl;
			std::cout << "Exiting" << std::endl;
			exit(1);
		}
		else{
			std::cout << "Loaded Left: " << left_img_filename << std::endl;
		}
		rFrame = imread(right_img_filename).getUMat(ACCESS_READ);
		if(lFrame.empty())
		{
			std::cout << "Failed to read right image \"" << right_img_filename << "\"" << std::endl;
			std::cout << "Exiting" << std::endl;
			exit(1);
		}
		else{
			std::cout << "Loaded Right: " << right_img_filename << std::endl;
		}
		gtFrameImg = imread(gt_img_filename).getUMat(ACCESS_READ);
		if(rFrame.empty()){
			std::cout << "Failed to read ground truth image \"" << gt_img_filename << "\"" << std::endl;
			std::cout << "Exiting" << std::endl;
			exit(1);
		}else{
			std::cout << "Loaded Ground Truth: " << gt_img_filename << std::endl;
		}
		//Pre-process ground truth image data
		cvtColor(gtFrameImg, gtFrame, CV_RGB2GRAY);
		minMaxLoc(gtFrame, &minVal_gt, &maxVal_gt);
		gtFrame.convertTo(gtFrame, CV_8U, 255/(maxVal_gt - minVal_gt));
		//Init error map
		eDispMap = UMat(lFrame.rows, lFrame.cols, CV_8UC1);
	}

#ifdef DISPLAY
	//#########################################################################
	//# Display Setup
	//#########################################################################
	//Set up display window to hold both input images and both output disparity maps
	//resizeWindow("InputOutput", lFrame.cols*2, lFrame.rows*2); //Rectified image size - not camera resolution size
#endif
	display_container = UMat(lFrame.rows*2, lFrame.cols*2, CV_8UC3);
	leftInputImg  = UMat(display_container, Rect(0,             0,           lFrame.cols, lFrame.rows)); //Top Left
	rightInputImg = UMat(display_container, Rect(lFrame.cols,   0,           lFrame.cols, lFrame.rows)); //Top Right
	leftDispMap   = UMat(display_container, Rect(0,             lFrame.rows, lFrame.cols, lFrame.rows)); //Bottom Left
	gtDispMap  = UMat(display_container, Rect(lFrame.cols,   lFrame.rows, lFrame.cols, lFrame.rows)); //Bottom Right
//	gtDispMap	  = UMat(display_container, Rect(lFrame.cols*2, 0, 			lFrame.cols, lFrame.rows)); //Top Far Right
//	errDispMap	  = UMat(display_container, Rect(lFrame.cols*2, lFrame.rows,	lFrame.cols, lFrame.rows)); //Bottom Far Right

	lFrame.copyTo(leftInputImg);
	rFrame.copyTo(rightInputImg);

	if(media_mode == DE_IMAGE){
		gtFrameImg.copyTo(gtDispMap);
	}

	//#########################################################################
    //# SGBM Mode Setup
    //#########################################################################
	//setupOpenCVSGBM(lFrame.channels(), maxDis);
	//imgDisparity16S = std::vector<cv::UMat>(num_threads, cv::UMat(lFrame.rows, lFrame.cols, CV_16S));
	blankDispMap = UMat(rFrame.rows, rFrame.cols, CV_8UC3);

	//#########################################################################
    //# GIF Mode Setup
    //#########################################################################
//	if(imgType == CV_32F){
		// Frame Preprocessing
//    cvtColor( lFrame, lFrame, CV_BGR2RGB );
//    cvtColor( rFrame, rFrame, CV_BGR2RGB );
//
//    lFrame.convertTo( lFrame_tmp, CV_32F, 1 / 255.0f );
//    rFrame.convertTo( rFrame_tmp, CV_32F,  1 / 255.0f );
//	}

//	SMDE = new DispEst(lFrame_tmp.getMat(ACCESS_READ), rFrame_tmp.getMat(ACCESS_READ), maxDis, num_threads, gotOCLDev);

	//#########################################################################
	//# End of Preprocessing (that we don't want to repeat)
    //#########################################################################
	//printf("End of Preprocessing\n");

    printf("StereoMatch Application Initialised\n");
	return;
}

StereoMatch::~StereoMatch(void)
{
	printf("Shutting down StereoMatch Application\n");
//	delete SMDE;
	if(media_mode == DE_VIDEO)
		cap.release();
	printf("Application Shut down\n");
}

//#############################################################################
//# Do stereo matching process
//#############################################################################
int StereoMatch::sgbm_thread(std::mutex &cap_m, std::mutex &dispMap_m, bool &end_de)
{
    //Frame Holders & Camera object
	cv::UMat lFrame_thread, rFrame_thread, vFrame_thread;
    cv::UMat lFrame_tmp_thread, rFrame_tmp_thread;
	cv::UMat lFrame_rec_thread, rFrame_rec_thread;

	//local disparity map containers
	cv::Ptr<StereoSGBM> ssgbm;
	setupOpenCVSGBM(ssgbm, lFrame.channels(), maxDis);
    cv::UMat imgDisparity16S = cv::UMat(lFrame.rows, lFrame.cols, CV_16S);

    cv::UMat lDispMap_thread, outDispMap;
	double minVal, maxVal;
	unsigned long long start_time, end_time;

    while(!end_de)
    {
        start_time = get_timestamp();
        //#########################################################################
        //# Frame Capture and Preprocessing (that we have to repeat)
        //#########################################################################
        if(media_mode == DE_VIDEO)
        {
#ifdef DEBUG_APP
            std::cout << "media_mode == DE_VIDEO" << std::endl;
#endif // DEBUG_APP

            cap_m.lock();
            cap >> vFrame_thread; //capture a frame from the camera
            cap_m.unlock();

            if(vFrame_thread.empty())
            {
                printf("Could not load camera frame\n");
                return -1;
            }

            lFrame_thread = vFrame_thread(Rect(0,0, vFrame_thread.cols/2,vFrame_thread.rows)); //split the frame into left
            rFrame_thread = vFrame_thread(Rect(vFrame_thread.cols/2, 0, vFrame_thread.cols/2, vFrame_thread.rows)); //and right images

            if(lFrame_thread.empty() || rFrame_thread.empty())
            {
                printf("No data in left or right frames\n");
                return -1;
            }

            //Applies a generic geometrical transformation to an image.
            //http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#remap
            remap(lFrame_thread, lFrame_rec_thread, mapl[0], mapl[1], INTER_LINEAR);
            remap(rFrame_thread, rFrame_rec_thread, mapr[0], mapr[1], INTER_LINEAR);

            lFrame_thread = lFrame_rec_thread(cropBox);
            rFrame_thread = rFrame_rec_thread(cropBox);

        }
        else{
			lFrame.copyTo(lFrame_thread);
			rFrame.copyTo(rFrame_thread);
        }

        //#########################################################################
        //# Start of Disparity Map Creation
        //#########################################################################

        if((lFrame_thread.type() & CV_MAT_DEPTH_MASK) != CV_8U){
            lFrame_thread.convertTo(lFrame_tmp_thread, CV_8U, 255);
            rFrame_thread.convertTo(rFrame_tmp_thread, CV_8U, 255);
        }
        else{
            lFrame_tmp_thread = lFrame_thread;
            rFrame_tmp_thread = rFrame_thread;

        }

        ssgbm->compute(lFrame_tmp_thread, rFrame_tmp_thread, imgDisparity16S); //Compute the disparity map
        minMaxLoc(imgDisparity16S, &minVal, &maxVal); //Check its extreme values
        imgDisparity16S.convertTo(lDispMap_thread, CV_8U, 255/(maxVal - minVal));
        //Load the disparity map to the display
        cvtColor(lDispMap_thread, outDispMap, CV_GRAY2RGB);

		end_time = get_timestamp();

        dispMap_m.lock();
#ifdef DISPLAY
		if(media_mode == DE_VIDEO)
		{
			lFrame_thread.copyTo(leftInputImg);
			rFrame_thread.copyTo(rightInputImg);
		}
		cv::applyColorMap(outDispMap, outDispMap, COLORMAP_JET);
		outDispMap.copyTo(leftDispMap);
#endif
		frame_rates.push_back((double)1000000/(end_time - start_time));
        dispMap_m.unlock();
    }
	return 0;
}


//#############################################################################
//# Calibration and Paramter loading for stereo camera setup
//#############################################################################
int StereoMatch::stereoCameraSetup(void)
{
	if(cap.get(CAP_PROP_FRAME_HEIGHT) != 376){
		cap.set(CAP_PROP_FRAME_HEIGHT, 376); //  376  (was 480),  720, 1080, 1242
		cap.set(CAP_PROP_FRAME_WIDTH, 1344); // 1344 (was 1280), 2560, 3840, 4416
		cap.set(CAP_PROP_FPS, 30); //376: {15, 30, 60, 100}, 720: {15, 30, 60}, 1080: {15, 30}, 1242: {15},

		if(cap.get(CAP_PROP_FRAME_HEIGHT) != 376){
			printf("Could not set correct frame resolution:\n");
			cout << "Target Height: " << 376 << endl;
			cout << "CAP_PROP_FRAME_HEIGHT: " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
			cout << "CAP_PROP_FRAME_WIDTH: " << cap.get(CAP_PROP_FRAME_WIDTH) << endl;
			exit(1);
		}
	}
	cout << "CAP_PROP_FPS: " << cap.get(CAP_PROP_FPS) << endl;
	cout << "CAP_PROP_FRAME_HEIGHT: " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
	cout << "CAP_PROP_FRAME_WIDTH: " << cap.get(CAP_PROP_FRAME_WIDTH) << endl;

	cap >> vFrame;
	if(vFrame.empty())
	{
		printf("Could not load camera frame\n");
		exit(1);
	}

	lFrame = vFrame(Rect(0,0, vFrame.cols/2,vFrame.rows));
	rFrame = vFrame(Rect(vFrame.cols/2, 0, vFrame.cols/2, vFrame.rows));
	if(lFrame.empty() || rFrame.empty())
	{
		printf("No data in left or right frames\n");
		exit(1);
	}

	if(recalibrate)
    {
		resizeWindow("InputOutput", lFrame.cols*2, lFrame.rows*2); //Native camera image resolution
		display_container = UMat(lFrame.rows*2, lFrame.cols*2, CV_8UC3);
		leftInputImg  = UMat(display_container, Rect(0,           0,           lFrame.cols, lFrame.rows)); //Top Left
		rightInputImg = UMat(display_container, Rect(lFrame.cols, 0,           lFrame.cols, lFrame.rows)); //Top Right
		leftDispMap   = UMat(display_container, Rect(0,           lFrame.rows, lFrame.cols, lFrame.rows)); //Bottom Left
		rightDispMap  = UMat(display_container, Rect(lFrame.cols, lFrame.rows, lFrame.cols, lFrame.rows)); //Bottom Right

		lFrame.copyTo(leftInputImg);
		rFrame.copyTo(rightInputImg);
		imshow("InputOutput", display_container);

		//###########################################################################################################
		//# Camera Calibration - find intrinsic & extrinsic parameters from first principles (contains rectification)
		//###########################################################################################################
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
		//#####################################################################
		//# Camera Setup - load existing intrinsic & extrinsic parameters
		//#####################################################################
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

//#############################################################################
//# Chessboard image capture for camera calibration
//#############################################################################
int StereoMatch::captureChessboards(void)
{
    int img_num = 0;
	char imageLoc[100];

	while(img_num < 10)
	{
		cap >> vFrame;

		if(vFrame.empty())
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
        printf("Usage: ./PRiMEStereoMatch VIDEO ( [RECALIBRATE | RECAL] [RECAPTURE | RECAP] )\n" );
        printf("Usage: \t or\n");
        printf("Usage: ./PRiMEStereoMatch IMAGE left_image_filename right_image_filename\n" );
		exit(1);
	}
	printf("Matching Algorithm Type: %s.\n", MatchingAlgorithm ? "STEREO_GIF" : "STEREO_SGBM");

	if(!strcmp(argv[1],"VIDEO")){
		printf("Media Type: \t\t Video Processing from Stereo Camera.\n");
		media_mode = DE_VIDEO;
		recalibrate = false;
		if(argc > 2){
			if(!strcmp(argv[2],"RECALIBRATE") || !strcmp(argv[2],"RECAL")) {
				printf("Recalibrating...\n");
				recalibrate = true;
			}
		}
		recaptureChessboards = false;
		if(argc > 3){
			if(!strcmp(argv[3],"RECAPTURE") || !strcmp(argv[3],"RECAP")) {
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
			printf("Usage: ./PRiMEStereoMatch IMAGE left_image_filename right_image_filename\n" );
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
				left_img_filename = std::string(argv[2]);
				std::cout << "Left Image: " << left_img_filename << std::endl;
			}
			else
			{
				printf("Left Image: Incompatible image filename extension specified. \nPlease use either .png, .ppm .jpg images\n");
				printf("Usage: ./PRiMEStereoMatch IMAGE left_image_filename right_image_filename\n" );
				exit(1);
			}

			img_ext = &argv[3][strlen(argv[3])-4];
			if(!strcmp(img_ext, ".png") || !strcmp(img_ext, ".jpg") || !strcmp(img_ext, ".ppm"))
			{
				right_img_filename = std::string(argv[3]);
				std::cout << "Right Image: " << right_img_filename << std::endl;
			}
			else
			{
				printf("Right Image: Incompatible image filename extension specified. \nPlease use either .png or jpg images\n");
				printf("Usage: ./PRiMEStereoMatch IMAGE left_image_filename right_image_filename\n" );
				exit(1);
			}
			media_mode = DE_IMAGE;
		}
	}
	else{
		printf("Invalid media type chosen:\n");
		printf("Usage: ./PRiMEStereoMatch [MEDIA TYPE = VIDEO|IMAGE [image_filenames]] ([RECALIBRATE | RCAL] [RECAPTURE | RCAP])\n" );
		exit(1);
	}
	return 0;
}

//#############################################################################################
//# Setup for OpenCV implementation of Stereo matching using Semi-Global Block Matching (SGBM)
//#############################################################################################
int StereoMatch::setupOpenCVSGBM(cv::Ptr<StereoSGBM>& ssgbm, int channels, int ndisparities)
{
	int mindisparity = 0;
	int SADWindowSize = 5;

	// Call the constructor for StereoSGBM
	ssgbm = cv::StereoSGBM::create(
		mindisparity, 								//minDisparity = 0,
		ndisparities, 								//numDisparities = 16,
		SADWindowSize, 								//blockSize = 3,
		8*channels*SADWindowSize*SADWindowSize, 	//P1 = 0,
		32*channels*SADWindowSize*SADWindowSize, 	//P2 = 0,
		1, 											//disp12MaxDiff = 0,
		63, 										//preFilterCap = 0,
		10, 										//uniquenessRatio = 0,
		100, 										//speckleWindowSize = 0,
		32, 										//speckleRange = 0,
		StereoSGBM::MODE_HH 						//mode = StereoSGBM::MODE_SGBM
		);

    return 0;
}
