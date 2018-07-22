/************************************************************************/
/*  Author: Yuan Yuan
/*  Date:2018/03/08
/*	Contact Information: Sichuan University, No.24 South Section 1, Yihuan Road, Chengdu , China, 610065
/*	Description: This is research code for the paper: A Fast Single-Image Super-Resolution Method Implemented With CUDA.
/*	More information can be found in readme.md
/*  If you find this code is useful or helpful to your own project,please cite:
/*	
/*  Email:1092743695@qq.com
/************************************************************************/

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>
#include <Windows.h>  //for high performance time counter
using namespace std;
//OpenCV
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
using namespace cv;

#include "cuSSSR.h"

int main(const int argc,const char** argv)
{
		if(argc < 2)
		{
			cout<<"Parameter Error."<<endl;
			system("pause");
			return -1;

		}
		//for high performance time counter
		LARGE_INTEGER t_Start,t_End,CPU_Frequency;
		//set parameters
		Parameters param;
		param.PatchSize = 5;	//patch size(pixels)
		param.SlidingLength = 1;//sliding length(pixels) of image patches
		param.K = 10;			//K most similar patches in search space
		param.ZoomCount = 2;	//level numbers of multi-level amplification
		param.UpFactor = 4;		//SR factor
		param.pQueryLocation = NULL;
		param.pQueryLocation = NULL;
		param.pReferHFPatches = NULL;
		param.pReferMFPatches = NULL;
		param.pKNNDist = NULL;
		param.pKNNIndex = NULL;

		string ImagePath = argv[1];
		string SavePath = ImagePath.substr(0,ImagePath.length()-4)+"_result.bmp";
		//read ground truth image
		Mat GTImage = imread(ImagePath,CV_LOAD_IMAGE_ANYDEPTH);
		GTImage.convertTo(GTImage,CV_32FC1);  //convert data type to float 
		if(GTImage.data != NULL)
		{
			//generate low resolution(LR) image by down sampling the ground truth image with Bi-cubic interpolation
			Mat InputLR;
			resize(GTImage,InputLR,Size(),1.0/param.UpFactor,1.0/param.UpFactor,CV_INTER_CUBIC);  

			param.FinalHeight = GTImage.rows;   
			param.FinalWidth = GTImage.cols;	
			//output HR Image
			Mat OutputHR;

			QueryPerformanceFrequency(&CPU_Frequency);
			QueryPerformanceCounter(&t_Start);

			//upscale LR image on GPU
			cuSSSR(InputLR,OutputHR,param);

			QueryPerformanceCounter(&t_End);
			float ElapsedTime = (float)(t_End.QuadPart - t_Start.QuadPart)/CPU_Frequency.QuadPart;
			cout<<"Run time: " <<ElapsedTime << " s"<<endl;
			//save the upscaled HR image
			imwrite(SavePath,OutputHR);

		}
		else
		{
			cout<<"Failed to open the image"<<endl;
		}
	system("pause");
	return 0;
}
