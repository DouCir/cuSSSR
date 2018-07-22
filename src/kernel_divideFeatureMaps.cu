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

#include <cuda.h>

/*
	kernel function : divide reference/query MF map into patches
*/
__global__ void kernel_divideMFMaps(float* dev_Map_1,float* dev_Map_2,float* dev_Map_3,int ImgWidth,int PatchSize,int Max_X,int Max_Y,int offset,int DataWidth,float* dev_PatchesData,float* dev_EnergyData)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; //horizontal direction
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; //vertical direction
	//unsigned int Max_X = (ImgWidth - PatchSize) / SlidingLength + 1;
	//unsigned int Max_Y = (ImgHeight - PatchSize) / SlidingLength + 1;
	if(x < Max_X && y < Max_Y)
	{
		int location = y*Max_X+x;
		int cnt = 0;
		int NV = PatchSize*PatchSize;
		float EnergySum = 0.0;
		//convert a matrix(PatchSize * PatchSize) into a column vector
		for(unsigned int m = y;m<y+PatchSize;m++)
		{
			for(unsigned int n = x;n<x+PatchSize;n++)
			{
				dev_PatchesData[cnt*DataWidth+location+offset] = dev_Map_1[m*ImgWidth+n];
				dev_PatchesData[(cnt+NV)*DataWidth+location+offset] = dev_Map_2[m*ImgWidth+n];
				dev_PatchesData[(cnt+2*NV)*DataWidth+location+offset] = dev_Map_3[m*ImgWidth+n];
				cnt++;

				EnergySum = EnergySum + abs(dev_Map_1[m*ImgWidth+n]) + abs(dev_Map_2[m*ImgWidth+n]) + abs(dev_Map_3[m*ImgWidth+n]);
			}
		}
		dev_EnergyData[location+offset] = EnergySum;
	}
}


/*
	kernel function : divide reference HF map into patches
*/
__global__ void kernel_divideHFMaps(float* dev_ImageData,int ImgWidth,int PatchSize,int Max_X,int Max_Y,int offset,int DataWidth,float* dev_PatchesData)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; //horizontal direction
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; //vertical direction
	//unsigned int Max_X = (ImgWidth - PatchSize) / SlidingLength + 1;
	//unsigned int Max_Y = (ImgHeight - PatchSize) / SlidingLength + 1;
	if(x < Max_X && y < Max_Y)
	{
		int location = y*Max_X+x;
		int cnt = 0;
		//convert a matrix(PatchSize * PatchSize) into a column vector
		for(unsigned int m = y;m<y+PatchSize;m++)
		{
			for(unsigned int n = x;n<x+PatchSize;n++)
			{
				dev_PatchesData[cnt*DataWidth+location+offset] = dev_ImageData[m*ImgWidth+n];
				cnt++;
			}
		}
	}
}