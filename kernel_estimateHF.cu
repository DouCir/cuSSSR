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

#include <math.h>
#include <cuda.h>
/*
	kernel function : estimate HF details of output HR image
*/
__global__ void kernel_estimateHF(float* dev_RefHF,float* device_KNNDist,int* device_KNNIndex,int* dev_location,float* Val,int PatchSize,int K,int ImgWidth,int ImgHeight,int ReferWidth,int QueryWidth,float* device_EstimatedHF,int* device_OverlapArea)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; //1D Block
	if(x < QueryWidth)
	{
		float* Weight = new float[K];
		//int SingleValSize = PatchSize*PatchSize*K;
		//float* Val = new float[PatchSize*PatchSize*K];
		float sumWeight = 0.0;
		double deltaD = 0.48*PatchSize*PatchSize;
		int OneValSize = PatchSize*PatchSize*K;
		//Calculate weight coefficient based on distance
		for(unsigned int k = 0;k<K;k++)
		{
			Weight[k] = exp((-1) * pow(device_KNNDist[k*QueryWidth+x],2) / (2 * pow(deltaD,2) ));
			sumWeight = sumWeight + Weight[k];
			for (unsigned int cnt_x = 0;cnt_x<PatchSize*PatchSize;cnt_x++)
			{
				Val[x*OneValSize+cnt_x * K + k] = dev_RefHF[cnt_x*ReferWidth+device_KNNIndex[k*QueryWidth+x]-1];  //最近的K个图像块
			}
		}
		float *Res = new float[PatchSize*PatchSize];
		//Normalization of weights
		for (unsigned int k = 0;k<K;k++)
		{
			Weight[k] = Weight[k]/sumWeight;
		}
		//estimate HF details
		for (unsigned int cnt_x = 0;cnt_x<PatchSize*PatchSize;cnt_x++)
		{
			Res[cnt_x] = 0.0;
			for(unsigned int k = 0;k<K;k++)
			{
				Res[cnt_x] = Weight[k]*Val[x*OneValSize + cnt_x*K+k]+Res[cnt_x];
			}
		}
		int cnt = 0;
		//record HF detials and overlap area 
		int row = dev_location[x*2];
		int col = dev_location[x*2+1];
		for(unsigned int m = row;m<row+PatchSize;m++)
		{
			for(unsigned int n = col;n<col+PatchSize;n++)
			{
				atomicAdd(&device_EstimatedHF[m*ImgWidth+n],Res[cnt++]);
				atomicAdd(&device_OverlapArea[m*ImgWidth+n],1);
			}
		}
		delete []Weight;
		delete []Res;
	}
}

/*
	kernel function : process the overlap area 
*/
__global__ void kernel_processOverlapArea(float* device_EstimatedHF,int* device_OverlapArea,int ImgWidth,int ImgHeight,int PatchSize,float* dev_Average)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x < ImgWidth && y < ImgHeight)
	{
		if(device_OverlapArea[y*ImgWidth+x] <= 0)
		{
			device_OverlapArea[y*ImgWidth+x] = 1;
		}
		dev_Average[y*ImgWidth+x] = device_EstimatedHF[y*ImgWidth+x]/device_OverlapArea[y*ImgWidth+x];
	}
}
