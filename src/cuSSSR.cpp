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


#include "cuSSSR.h"
#include <math.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
using namespace std;

//declare CUDA functions
/*
	the wrap of dividing reference feature maps into patches on GPU
*/
extern void wrap_divideReferMFMaps(vector<Mat*> & vReferMFMaps,Parameters & param);

/*
	the wrap of dividing query feature maps into patches on GPU
*/
extern void wrap_divideQueryMFMaps(vector<Mat*> & vQueryMFMaps,Parameters & param);

/*
	the wrap of dividing high frequency maps into patches on GPU
*/
extern void wrap_divideHFMaps(vector<Mat* >& vHFMaps,Parameters & param);

/*
	the wrap of KNN search
*/
extern void wrap_KNN(float* ref_host, int ref_width, float* query_host, int query_width, int height, int k, float* dist_host, int* ind_host);

/*
	the wrap of estimating HR image
*/
extern void wrap_estimateHR(const Mat & InputIntpHR,Mat & OutputHR,Parameters &param);

/*
	function : cuSSSR:self-similarity based SR method implemented with CUDA
*/
void cuSSSR(Mat & InputLR,Mat & OutputHR,Parameters & param)
{
	//calculate zoom factor used in multi-level amplification procedure
	param.ZoomFactor = pow(param.UpFactor,1.0/param.ZoomCount);
	//intermediate level
	Mat TempOutputHR;
	for (unsigned int i = 0;i<param.ZoomCount-1;i++)
	{
		SSSR(InputLR,TempOutputHR,param);
		InputLR = TempOutputHR;
	}
	//final stage
	param.ZoomFactor = -1.0;
	if (param.ZoomCount > 1)
		SSSR(TempOutputHR,OutputHR,param);
	else
		SSSR(InputLR,OutputHR,param);
}
/*
	function : single amplification in multi-level amplification
*/
void SSSR(const Mat & InputLR,Mat & OutputHR,Parameters & param)
{
	TrainingPhase(InputLR,param);
	InferringPhase(InputLR,OutputHR,param);
	//free memory
	if(param.pKNNDist != NULL) free(param.pKNNDist);
	if(param.pKNNIndex != NULL) free(param.pKNNIndex);
	if(param.pReferHFPatches != NULL) free(param.pReferHFPatches);
	if(param.pReferMFPatches != NULL) free(param.pReferMFPatches);
	if(param.pQueryLocation != NULL) free(param.pQueryLocation);
	if(param.pQueryMFPatches != NULL) free(param.pQueryMFPatches);
	param.pKNNDist = NULL;
	param.pKNNIndex = NULL;
	param.pReferHFPatches = NULL;
	param.pReferMFPatches = NULL;
	param.pQueryLocation = NULL;
	param.pQueryMFPatches = NULL;
	vector<int>().swap(param.vReferLocation);
}
/*
	function : the training phase : build training set by taking advantages of self-similarity of images
*/
void TrainingPhase(const Mat & InputLR,Parameters & param)
{
	//step 1 - step 5
	vector<Mat*> vReferHFMaps;  //reference HF maps
	vector<Mat*> vReferMFMaps; // reference MF maps
	TrainingPhase_getMaps(InputLR,vReferMFMaps,vReferHFMaps,param);
	//launch kernel function to divide MF maps and HF maps into little patches in parallel on GPU
	// step 6
	wrap_divideReferMFMaps(vReferMFMaps,param);
	// step 7
	wrap_divideHFMaps(vReferHFMaps,param);
}

/*
	function : the inferring phase : estimate HF details and generate HR image
*/
void InferringPhase(const Mat & InputLR,Mat & OutputHR,Parameters & param)
{
	// step 8 : enlarge input LR image by Bi-cubic interpolation
	vector<Mat*> vQueryMFMaps;	// the query MF maps
	Mat HR_Intp;				// interpolated HR image
	if (param.ZoomFactor < 0.0)
		resize(InputLR,HR_Intp,Size(param.FinalWidth,param.FinalHeight),0.0,0.0,CV_INTER_CUBIC);
	else
		resize(InputLR,HR_Intp,Size(0,0),param.ZoomFactor,param.ZoomFactor,CV_INTER_CUBIC);

	// step 9 : get MF maps by exploiting DoG filters
	InferingPhase_getMaps(HR_Intp,vQueryMFMaps,param);

	//step 10 : divide query MF maps into little patches
	//launch kernel function to divide query MF maps into little patches On GPU
	wrap_divideQueryMFMaps(vQueryMFMaps,param);
	
	// step 11 : search K most similar reference patches in reference patches for each query patch 
	param.pKNNIndex = (int*)malloc(param.QueryWidth * param.K * sizeof(int));
	param.pKNNDist = (float*)malloc(param.QueryWidth * param.K * sizeof(float));
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsed_time;
	cudaEventRecord(start, 0);
	wrap_KNN(param.pReferMFPatches,param.ReferWidth,param.pQueryMFPatches,param.QueryWidth,param.PatchSize*param.PatchSize*3,param.K,param.pKNNDist,param.pKNNIndex);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("Searching k-Nearest Neighbor in %f s \n", elapsed_time/1000);

	/* step 12 - step 14: estimation HF details */

	wrap_estimateHR(HR_Intp,OutputHR,param);
}
/*
	function : get reference MF maps and HF maps
*/
void TrainingPhase_getMaps(const Mat & InputLR,vector<Mat*> & vReferMFMaps,vector<Mat*> & vReferHFMaps,Parameters &param)
{
	Mat Im_LL;				  // down-sample version of input LR image
	Mat Im_LH;				  // interpolated image of down-sample version image
	Mat dstG1,dstG2;		  // the feature images obtained by DoG filter
	Point2f center = Point2f(InputLR.cols / 2, InputLR.rows / 2);	// the center of rotation											
	double scale = 1;												// zoom scale(no zoom here)
	Mat rotationMat;		// rotation mat
	Mat rotatedImg;			// rotated image
	/* step 1 - step 5 */
	for(int i = 0;i<4;i++)
	{
		double angle = double(90*i);	// the rotation angle
		rotationMat = getRotationMatrix2D(center, angle, scale);
		warpAffine(InputLR, rotatedImg, rotationMat, InputLR.size(),INTER_CUBIC); 
		//Mat BluredImg;
		//GaussianBlur(rotateImg,BluredImg,Size(PatchSize,PatchSize),0.5,0.5);  //gaussian blur
		//down-sample
		if(param.ZoomFactor > 0.0)		//intermediate level
			resize(rotatedImg,Im_LL,Size(0,0),1.0/param.ZoomFactor,1.0/param.ZoomFactor,CV_INTER_CUBIC);
		else							//final level
			resize(rotatedImg,Im_LL,Size(0,0),double(InputLR.rows)/param.FinalHeight,double(InputLR.rows)/param.FinalHeight,CV_INTER_CUBIC);
		//upscale
		resize(Im_LL,Im_LH,rotatedImg.size(),0.0,0.0,CV_INTER_CUBIC);			// Im_LL/Im_LH : LR/HR image pair
		//obtain HF maps by exploiting differential operation between input LR image and enlarged image
		Mat* HFMap = new Mat(rotatedImg - Im_LH);
		vReferHFMaps.push_back(HFMap);
		//exploit multi-scale DoG filters to obtain multi-view MF maps
		GaussianBlur(Im_LH,dstG1,Size(param.PatchSize,param.PatchSize),0.9,0.9,BORDER_REFLECT); // first scale DoG filter
		GaussianBlur(Im_LH,dstG2,Size(param.PatchSize,param.PatchSize),0.6,0.6,BORDER_REFLECT);
		Mat* dstDoG_1 = new Mat(dstG1 - dstG2);
		vReferMFMaps.push_back(dstDoG_1);

		GaussianBlur(Im_LH,dstG1,Size(param.PatchSize,param.PatchSize),0.8,0.8,BORDER_REFLECT); // second scale DoG filter
		GaussianBlur(Im_LH,dstG2,Size(param.PatchSize,param.PatchSize),0.5,0.5,BORDER_REFLECT);
		Mat* dstDoG_2 = new Mat(dstG1-dstG2);
		vReferMFMaps.push_back(dstDoG_2);

		GaussianBlur(Im_LH,dstG1,Size(param.PatchSize,param.PatchSize),0.7,0.7,BORDER_REFLECT); // third scale DoG filter
		GaussianBlur(Im_LH,dstG2,Size(param.PatchSize,param.PatchSize),0.4,0.4,BORDER_REFLECT);
		Mat* dstDoG_3 =  new Mat(dstG1-dstG2);
		vReferMFMaps.push_back(dstDoG_3);
	}
}

/*
	function : get query MF maps 
*/
void InferingPhase_getMaps(const Mat & HR_Intp,vector<Mat*> & vQueryMFMaps,Parameters &param)
{
	// step 9 : obtain query MF maps and query patches
	Mat dstG1,dstG2;	// the feature images obtained by DoG filter

	GaussianBlur(HR_Intp,dstG1,Size(param.PatchSize,param.PatchSize),0.9,0.9,BORDER_REFLECT); // first scale DoG filter
	GaussianBlur(HR_Intp,dstG2,Size(param.PatchSize,param.PatchSize),0.6,0.6,BORDER_REFLECT);
	Mat* dstDoG_1 = new Mat(dstG1 - dstG2);
	vQueryMFMaps.push_back(dstDoG_1);

	GaussianBlur(HR_Intp,dstG1,Size(param.PatchSize,param.PatchSize),0.8,0.8,BORDER_REFLECT); // second scale DoG filter
	GaussianBlur(HR_Intp,dstG2,Size(param.PatchSize,param.PatchSize),0.5,0.5,BORDER_REFLECT);
	Mat* dstDoG_2  = new Mat(dstG1 - dstG2);
	vQueryMFMaps.push_back(dstDoG_2);

	GaussianBlur(HR_Intp,dstG1,Size(param.PatchSize,param.PatchSize),0.7,0.7,BORDER_REFLECT); // third scale DoG filter
	GaussianBlur(HR_Intp,dstG2,Size(param.PatchSize,param.PatchSize),0.4,0.4,BORDER_REFLECT);
	Mat* dstDoG_3 = new Mat(dstG1 - dstG2);
	vQueryMFMaps.push_back(dstDoG_3);
}

