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


#ifndef _H_CUSSSR
#define _H_CUSSSR

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
using namespace cv;

typedef struct {
	unsigned int PatchSize;		//image patch size(pixels)
	unsigned int SlidingLength;	//sliding length(pixels)
	unsigned int FinalWidth;	//the width of final HR image
	unsigned int FinalHeight;	//the height of final HR image
	unsigned int K;				//the numbers of most similar patches
	unsigned int ZoomCount;		//the numbers of levels in multi-level amplification
	double UpFactor;			//total zoom factor
	double ZoomFactor;			//zoom factor in intermediate level
	float EnergyThreshold;		//energy threshold line used to filtering MF patches
	float* pReferMFPatches;		//reference MF patches
	float* pReferHFPatches;		//reference HF patches
	float* pQueryMFPatches;		//query MF patches
	int* pKNNIndex;				//the indices of K most similar patches in reference space
	float* pKNNDist;			//the distance between query patches and K most similar patches in reference space
	vector<int> vReferLocation; //the location of reference patches that over the energy threshold
	int* pQueryLocation;		//the location of query patches that over the energy threshold
	int ReferWidth;				//the number of reference patches
	int QueryWidth;				//the number of query patches
}Parameters;


//************************************
// Abstract: the API of our paper
// FullName:  cuSSSR
// Access:    public 
// Returns:   void
// Qualifier:
// Parameter: Mat & InputLR
// Parameter: Mat & OutputHR
// Parameter: Parameters & param
//************************************
void cuSSSR(Mat & InputLR,Mat & OutputHR,Parameters & param);

//************************************
// Abstract: single amplification in multi-level amplification
// FullName:  SSSR
// Access:    public 
// Returns:   void
// Qualifier:
// Parameter: const Mat & InputLR
// Parameter: Mat & OutputHR
// Parameter: Parameters & param
//************************************
void SSSR(const Mat & InputLR,Mat & OutputHR,Parameters & param);

//************************************
// Abstract:  Building training set by taking advantage of self-similarity in image(step 1 - step 7)
// FullName:  TrainingPhase
// Access:    public 
// Returns:   void
// Qualifier:
// Parameter: const Mat & InputLR
// Parameter: Parameters & param
//************************************
void TrainingPhase(const Mat & InputLR,Parameters & param);

//************************************
// Abstract:  Inferring high resolution image (step 8 - step 14)
// FullName:  InferringPhase
// Access:    public 
// Returns:   void
// Qualifier:
// Parameter: const Mat & InputLR
// Parameter: Mat & OutputHR
// Parameter: Parameters & param
//************************************
void InferringPhase(const Mat & InputLR,Mat & OutputHR,Parameters & param);

//************************************
// Abstract:  get HF maps and MF maps during training phase
// FullName:  TrainingPhase_getMaps
// Access:    public 
// Returns:   void
// Qualifier:
// Parameter: const Mat & InputLR
// Parameter: vector<Mat * > vReferMFMaps
// Parameter: vector<Mat * > vReferHFMaps
// Parameter: Parameters & param
//************************************
void TrainingPhase_getMaps(const Mat & InputLR,vector<Mat*> & vReferMFMaps,vector<Mat*> & vReferHFMaps,Parameters &param);

//************************************
// Abstract:   get MF maps during training phase
// FullName:  InferingPhase_getMaps
// Access:    public 
// Returns:   void
// Qualifier:
// Parameter: const Mat & InputIntpHR
// Parameter: vector<Mat * > vQueryMFMaps
// Parameter: Parameters & param
//************************************
void InferingPhase_getMaps(const Mat & InputIntpHR,vector<Mat*> & vQueryMFMaps,Parameters &param);

#endif