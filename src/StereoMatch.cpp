/*---------------------------------------------------------------------------
   StereoMatch.cpp - Stereo Matching Application Code/Class
  ---------------------------------------------------------------------------
   Author: Charles Leech
   Email: cl19g10 [at] ecs.soton.ac.uk
   Copyright (c) 2016 Charlie Leech, University of Southampton.
   All rights reserved.
  ---------------------------------------------------------------------------*/
#include "StereoMatch.h"

//#############################################################################
//# SM Preprocessing that we don't want to repeat
//#############################################################################
StereoMatch::StereoMatch(int argc, const char *argv[], int gotOpenCLDev) :
	end_de(false)
{
    printf("Disparity Estimation for Depth Analysis in Stereo Vision Applications.\n");
    printf("Preprocessing for Stereo Matching.\n");

	//#########################################################################
    //# Setup - check input arguments
    //#########################################################################
	if(parse_cli(argc, argv))
		exit(1);

    unsigned int cam_height = 376; //  376  (was 480),  720, 1080, 1242
	unsigned int cam_width = 1344; // 1344 (was 1280), 2560, 3840, 4416

	maxDis = 64;
	de_mode = OCL_DE;
	//de_mode = OCV_DE;
	//num_threads = MIN_CPU_THREADS;
	num_threads = MAX_CPU_THREADS;
	gotOCLDev = gotOpenCLDev;
	error_threshold = 4 * (256/maxDis);

	if(media_mode == DE_VIDEO)
	{
		//#####################################################################
		//# Video/Camera Mode
		//#####################################################################
        cap = VideoCapture(0);
		if (!cap.isOpened()){
			printf("Could not open the VideoCapture device.\n");
			exit(1);
		}
		printf("Opened the VideoCapture device.\n");

		if (setCameraResolution(cam_height, cam_width)){
			printf("Could not set the camera resolution.\n");
			exit(1);
		}

		cap >> vFrame;
		if(vFrame.empty()){
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

		stereoCameraSetup();
	}
	else
	{
		//#####################################################################
		//# Image Mode
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
		//Pre-process grund truth image data
		cvtColor(gtFrameImg, gtFrame, CV_RGB2GRAY);
		minMaxLoc(gtFrame, &minVal_gt, &maxVal_gt);
		gtFrame.convertTo(gtFrame, CV_8U, 255/(maxVal_gt - minVal_gt));
		//Init error map
		eDispMap = UMat(lFrame.rows, lFrame.cols, CV_8UC1);
	}

	//#########################################################################
	//# Display Setup
	//#########################################################################
	//Set up display window to hold both input images and both output disparity maps
	display_container = UMat(lFrame.rows*2, lFrame.cols*3, CV_8UC3);
	leftInputImg  = UMat(display_container, Rect(0,             0,           lFrame.cols, lFrame.rows)); //Top Left
	rightInputImg = UMat(display_container, Rect(lFrame.cols,   0,           lFrame.cols, lFrame.rows)); //Top Right
	leftDispMap   = UMat(display_container, Rect(0,             lFrame.rows, lFrame.cols, lFrame.rows)); //Bottom Left
	rightDispMap  = UMat(display_container, Rect(lFrame.cols,   lFrame.rows, lFrame.cols, lFrame.rows)); //Bottom Right
	gtDispMap	  = UMat(display_container, Rect(lFrame.cols*2, 0, 			lFrame.cols, lFrame.rows)); //Top Far Right
	errDispMap	  = UMat(display_container, Rect(lFrame.cols*2, lFrame.rows,	lFrame.cols, lFrame.rows)); //Bottom Far Right

	lFrame.copyTo(leftInputImg);
	rFrame.copyTo(rightInputImg);

	if(media_mode == DE_IMAGE){
		gtFrameImg.copyTo(gtDispMap);
	}
#ifdef DISPLAY
	resizeWindow("InputOutput", display_container.cols, display_container.rows); //Rectified image size - not camera resolution size
#endif

	//#########################################################################
    //# SGBM Mode Setup
    //#########################################################################
	setupOpenCVSGBM(lFrame.channels(), maxDis);
	imgDisparity16S = UMat(lFrame.rows, lFrame.cols, CV_16S);
	blankDispMap = UMat(rFrame.rows, rFrame.cols, CV_8UC3);

	//#########################################################################
    //# GIF Mode Setup
    //#########################################################################
//	if(imgType == CV_32F){
		// Frame Preprocessing
    cvtColor( lFrame, lFrame, CV_BGR2RGB );
    cvtColor( rFrame, rFrame, CV_BGR2RGB );

    lFrame.convertTo( lFrame, CV_32F, 1 / 255.0f );
    rFrame.convertTo( rFrame, CV_32F,  1 / 255.0f );
//	}

	SMDE = new DispEst(lFrame.getMat(ACCESS_READ), rFrame.getMat(ACCESS_READ), maxDis, num_threads, gotOCLDev);

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
	delete SMDE;
	if(media_mode == DE_VIDEO)
		cap.release();
	printf("Application Shut down\n");
}

float get_rt(){
	struct timespec realtime;
	clock_gettime(CLOCK_MONOTONIC,&realtime);
	return (float)(realtime.tv_sec*1000000+realtime.tv_nsec/1000);
}

//#############################################################################
//# Complete GIF stereo matching process
//#############################################################################
void StereoMatch::compute(float& de_time_ms)
{
#ifdef DEBUG_APP
	std::cout << "Computing Depth Map" << std::endl;
#endif // DEBUG_APP

	float start_time = get_rt();
	//#########################################################################
	//# Frame Capture and Preprocessing (that we have to repeat)
	//#########################################################################
	if(media_mode == DE_VIDEO)
	{
		for(int drop=0;drop<30;drop++)
			cap >> vFrame; //capture a frame from the camera
		if(vFrame.empty())
		{
			printf("Could not load camera frame\n");
			return;
		}

		lFrame = vFrame(Rect(0,0, vFrame.cols/2,vFrame.rows)); //split the frame into left
		rFrame = vFrame(Rect(vFrame.cols/2, 0, vFrame.cols/2, vFrame.rows)); //and right images

		if(lFrame.empty() || rFrame.empty())
		{
			printf("No data in left or right frames\n");
			return;
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
#ifdef DEBUG_APP
	else if(media_mode == DE_IMAGE){
        std::cout << "media_mode == DE_IMAGE" << std::endl;
	}
	else {
        std::cout << "Unrecognised value for media_mode: " << media_mode << std::endl;
	}
#endif // DEBUG_APP

	//#########################################################################
	//# Start of Disparity Map Creation
	//#########################################################################
	if(MatchingAlgorithm == STEREO_SGBM)
	{
#ifdef DEBUG_APP
		printf("MatchingAlgorithm == STEREO_SGBM\n");
#endif // DEBUG_APP
		if((lFrame.type() & CV_MAT_DEPTH_MASK) != CV_8U){
			lFrame.convertTo(lFrame_tmp, CV_8U, 255);
			rFrame.convertTo(rFrame_tmp, CV_8U, 255);
		}

		ssgbm->compute(lFrame_tmp, rFrame_tmp, imgDisparity16S); //Compute the disparity map
		minMaxLoc(imgDisparity16S, &minVal, &maxVal); //Check its extreme values
		imgDisparity16S.convertTo(lDispMap, CV_8U, 255/(maxVal - minVal));
		//Load the disparity map to the display
		//applyColorMap( leftDispMap, leftDispMap, COLORMAP_JET);
		cvtColor(lDispMap, leftDispMap, CV_GRAY2RGB);
	}
	else if(MatchingAlgorithm == STEREO_GIF)
	{
#ifdef DEBUG_APP
		printf("MatchingAlgorithm == STEREO_GIF\n");
#endif // DEBUG_APP
		if((lFrame.type() & CV_MAT_DEPTH_MASK) != CV_8U)
        {
            cvtColor(lFrame, lFrame, CV_BGR2RGB);
            cvtColor(rFrame, rFrame, CV_BGR2RGB);
            lFrame.convertTo(lFrame_tmp, CV_32F, 1 / 255.0f);
            rFrame.convertTo(rFrame_tmp, CV_32F,  1 / 255.0f);

            SMDE->lImg = lFrame_tmp.getMat(ACCESS_READ);
            SMDE->rImg = rFrame_tmp.getMat(ACCESS_READ);
		}
		SMDE->threads = num_threads;

		// ******** Disparity Estimation Code ******** //
#ifdef DEBUG_APP
		std::cout <<  "Disparity Estimation Started..." << std::endl;
#endif // DEBUG_APP

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
		//cv::applyColorMap( SMDE->lDisMap*4, lDispMap, COLORMAP_JET); // *4 for conversion from disparty range (0-64) to RGB char range (0-255)
		cv::minMaxLoc(SMDE->lDisMap, &minVal, &maxVal);
		SMDE->lDisMap.convertTo(lDispMap, CV_8U, 4);//255/(maxVal - minVal));
		cv::cvtColor(lDispMap, leftDispMap, CV_GRAY2RGB);

		//cv::applyColorMap( SMDE->rDisMap*4, rDispMap, COLORMAP_JET);
		SMDE->rDisMap.convertTo(rDispMap, CV_8U, 255/(maxVal - minVal));
		cv::cvtColor(rDispMap, rightDispMap, CV_GRAY2RGB);
		// ******** Show Disparity Map  ******** //

#ifdef DEBUG_APP
		printf("CVC Time:\t | %8.2f ms\n",cvc_time/1000);
		printf("CVF Time:\t | %8.2f ms\n",cvf_time/1000);
		printf("DispSel Time:\t | %8.2f ms\n",dispsel_time/1000);
		printf("PP Time:\t | %8.2f ms\n",pp_time/1000);
#endif
	}

	de_time_ms = (get_rt() - start_time)/1000;
	printf("DE Time:\t | %8.2f ms\n", de_time_ms);

#ifdef DEBUG_APP
	cv::imwrite("leftDisparityMap.png", leftDispMap.getMat(ACCESS_READ));
	cv::imwrite("rightDisparityMap.png", rightDispMap.getMat(ACCESS_READ));
#endif

	if(media_mode == DE_IMAGE)
	{
		//Check pixel errors against ground truth depth map here.
		//Can only be done with images as golden reference is required.
		float num_bad_pixels = 0;
		float avg_err = 0;

		Mat eDispMap_Mat = eDispMap.getMat(ACCESS_READ);
		Mat lDispMap_Mat = lDispMap.getMat(ACCESS_READ);
		Mat gtFrame_Mat = gtFrame.getMat(ACCESS_READ);

		for(int y = 0; y < gtFrame.rows; y++)
		{
			uchar* eData = (uchar*) eDispMap_Mat.ptr<uchar>( y );
			uchar* lData = (uchar*) lDispMap_Mat.ptr<uchar>( y );
			uchar* gtData = (uchar*) gtFrame_Mat.ptr<uchar>( y );

			for(int x = 0; x < maxDis; x++)
			{
				eData[x] = 0;
			}
			for(int x = maxDis; x < gtFrame.cols; x++)
			{
				eData[x] = abs(lData[x] - gtData[x]);
				avg_err += (float)eData[x];
				if(eData[x] > error_threshold){
					num_bad_pixels++;
				}
				else{
					eData[x] = 0;
				}
			}
		}
		float num_pixels = gtFrame.cols*gtFrame.rows;
#ifdef DEBUG_APP
		printf("%%BP = %.2f%% \t Avg Err = %.2f\n", (float)num_bad_pixels*100/num_pixels, (float)avg_err/num_pixels);
#endif

		minMaxLoc(eDispMap, &minVal, &maxVal);
		//eDispMap.convertTo(eDispMap, CV_8U, 255/(maxVal - minVal));
		cvtColor(eDispMap, errDispMap, CV_GRAY2RGB);
	}
	return;
}

//#############################################################################
//# Calibration and Parameter loading for stereo camera setup
//#############################################################################
int StereoMatch::setCameraResolution(unsigned int height, unsigned int width)
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
			return -1;
		}
	}
	cout << "CAP_PROP_FPS: " << cap.get(CAP_PROP_FPS) << endl;
	cout << "CAP_PROP_FRAME_HEIGHT: " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
	cout << "CAP_PROP_FRAME_WIDTH: " << cap.get(CAP_PROP_FRAME_WIDTH) << endl;

	return 0;
}

//#############################################################################
//# Calibration and Parameter loading for stereo camera setup
//#############################################################################
int StereoMatch::stereoCameraSetup(void)
{
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

int StereoMatch::parse_cli(int argc, const char * argv[])
{
    args::ArgumentParser parser("Application: Stereo Matching for Depth Estimation.","PRiME Project\n");
    parser.LongSeparator("=");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

    args::Group commands(parser, "Commands (1 must be specified):", args::Group::Validators::Xor);

    args::Command arg_video(commands, "video", "Use video as the input source.", [&](args::Subparser &s_parser)
    {
		args::Flag arg_recalibrate(s_parser, "RECALIBRATE", "Recalibrate the camera to find ROIs.", {"RECALIBRATE"});
		args::Flag arg_recapture(s_parser, "RECAPTURE", "Recapture chessboard image pairs for recalibration.", {"RECAPTURE"});

		s_parser.Parse();

        std::cout << "Input Source: Video" << std::endl;
        std::cout << "Parsing command-specific arguments." << std::endl;

		media_mode = DE_VIDEO;
		recalibrate = false;
		recaptureChessboards = false;
		if(arg_recalibrate){
			recalibrate = true;
			if(arg_recapture){
				recaptureChessboards = true;
			}
		}
    });

    args::Command arg_image(commands, "image", "Use images as the input source.", [&](args::Subparser &s_parser)
    {
		args::ValueFlag<std::string> arg_left(s_parser, "left", "Left image filename.", {'l', "left"}, args::Options::Required);
		args::ValueFlag<std::string> arg_right(s_parser, "right", "Right image filename.", {'r', "right"}, args::Options::Required);
		args::ValueFlag<std::string> arg_gt(s_parser, "gt", "Ground truth image filename.", {'g', "gt"}, args::Options::Required);

		s_parser.Parse();

        std::cout << "Input Source: Images" << std::endl;
        std::cout << "Parsing command-specific arguments." << std::endl;

		media_mode = DE_IMAGE;
        left_img_filename = args::get(arg_left);
        right_img_filename = args::get(arg_right);
        gt_img_filename = args::get(arg_gt);
    });

	args::Options ReqGlobal = args::Options::Required | args::Options::Global;
    args::ValueFlag<std::string> arg_alg_mode(parser, "alg. mode", "The stereo matching algorithm to use.", {'a', "algorithm"}, ReqGlobal);

    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help) {
		std::cerr << parser;
        return -1;
    } catch (args::ParseError e) {
        std::cerr << e.what() << std::endl;
		std::cerr << parser;
        return -1;
    } catch (args::ValidationError e) {
        std::cerr << e.what() << std::endl;
		std::cerr << parser;
        return -1;
    }

	std::cout << "Global arguments:" << std::endl;
    if(args::get(arg_alg_mode) == "STEREO_GIF"){
		MatchingAlgorithm = STEREO_GIF;
		std::cout << "Using STEREO_GIF" << std::endl;
    } else {
		MatchingAlgorithm = STEREO_SGBM;
		std::cout << "Using STEREO_SGBM" << std::endl;
	}

    return 0;
}

//#############################################################################################
//# Setup for OpenCV implementation of Stereo matching using Semi-Global Block Matching (SGBM)
//#############################################################################################
int StereoMatch::setupOpenCVSGBM(int channels, int ndisparities)
{
	int mindisparity = 0;
	int SADWindowSize = 5;

	// Call the constructor for StereoSGBM
	ssgbm = StereoSGBM::create(
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
