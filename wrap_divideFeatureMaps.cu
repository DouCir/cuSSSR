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

//standard library
#include <stdio.h>
#include <math.h>
#include <vector>
using namespace std;
//CUDA runtime library
#include <cuda.h>
#include <cuda_runtime.h>
//OpenCV library
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
using namespace cv;
//kernel functions
#include "kernel_divideFeatureMaps.cu"

#include "cuSSSR.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

extern void printfErrorInfo(const char* DisplayStr,const char* ErrString);

//************************************
// Abstract:  launch CUDA kernel function to divide reference feature maps(MF maps) into little patches
// FullName:  wrap_divideReferMFMaps
// Access:    public 
// Returns:   void
// Qualifier:
// Parameter: vector<Mat * > & vReferMFMaps
// Parameter: Parameters & param
//************************************
void wrap_divideReferMFMaps(vector<Mat*> & vReferMFMaps,Parameters & param)
{
	int size_float = sizeof(float);
	int Max_X = (vReferMFMaps[0]->cols - param.PatchSize) / param.SlidingLength + 1;	//max numbers of patches in horizontal direction
	int Max_Y = (vReferMFMaps[0]->rows - param.PatchSize) / param.SlidingLength + 1;	//max numbers of patches in vertical direction
	int offset = Max_X * Max_Y;
	int PatchesWidth = 4 * offset;	//total patches
	int GridWidth,GridHeight;		//grid dimension 
	int FeatureDim = param.PatchSize * param.PatchSize * 3;
	size_t size_PatchesData = PatchesWidth * FeatureDim * size_float;
	size_t size_DoGMap = vReferMFMaps[0]->rows * vReferMFMaps[0]->cols * size_float;
	dim3 DimBlock(BLOCK_SIZE,BLOCK_SIZE);

	float* device_DoGMap1 = NULL;
	float* device_DoGMap2 = NULL;
	float* device_DoGMap3 = NULL;
	float* device_PatchesData = NULL;
	float* device_PatchesEnergy = NULL;
	cudaError_t err = cudaSuccess;
	err = cudaMalloc((void **)&device_DoGMap1,size_DoGMap);	//allocate GPU global memory
	if (err != cudaSuccess) printfErrorInfo("cudaMalloc Error at [device_DoGMap1]",cudaGetErrorString(err));  

	err = cudaMalloc((void **)&device_DoGMap2,size_DoGMap);
	if (err != cudaSuccess) printfErrorInfo("cudaMalloc Error at [device_DoGMap2]",cudaGetErrorString(err));  

	err = cudaMalloc((void **)&device_DoGMap3,size_DoGMap);
	if (err != cudaSuccess) printfErrorInfo("cudaMalloc Error at [device_DoGMap3]",cudaGetErrorString(err));  

	err = cudaMalloc((void **)&device_PatchesData,size_PatchesData);
	if (err != cudaSuccess) printfErrorInfo("cudaMalloc Error at [device_PatchesData]",cudaGetErrorString(err));  

	err = cudaMalloc((void **)&device_PatchesEnergy,PatchesWidth*size_float);
	if (err != cudaSuccess) printfErrorInfo("cudaMalloc Error at [device_PatchesEnergy]",cudaGetErrorString(err));  
	for (int i = 0;i < 4; i++)
	{
		//copy data from CPU to GPU
		err = cudaMemcpy(device_DoGMap1,(float*)vReferMFMaps[i*3]->data,size_DoGMap,cudaMemcpyHostToDevice);
		if (err != cudaSuccess) printfErrorInfo("cudaMemcpy Error occur when copying vReferMFMaps[i*3]->data to device_DoG1",cudaGetErrorString(err));

		err = cudaMemcpy(device_DoGMap2,(float*)vReferMFMaps[i*3+1]->data,size_DoGMap,cudaMemcpyHostToDevice);
		if (err != cudaSuccess) printfErrorInfo("cudaMemcpy Error occur when copying vReferMFMaps[i*3+1]->data to device_DoG2",cudaGetErrorString(err));

		err = cudaMemcpy(device_DoGMap3,(float*)vReferMFMaps[i*3+2]->data,size_DoGMap,cudaMemcpyHostToDevice);
		if (err != cudaSuccess) printfErrorInfo("cudaMemcpy Error occur when copying vReferMFMaps[i*3+2]->data to device_DoG3",cudaGetErrorString(err));

		int ImgWidth = vReferMFMaps[i*3]->cols;
		int ImgHeight = vReferMFMaps[i*3]->rows;
		int Max_X = (ImgWidth - param.PatchSize) / param.SlidingLength + 1;
		int Max_Y = (ImgHeight - param.PatchSize)/ param.SlidingLength + 1;
		GridWidth = (Max_X + BLOCK_SIZE - 1) / BLOCK_SIZE;
		GridHeight = (Max_Y + BLOCK_SIZE - 1)/ BLOCK_SIZE;
		dim3 DimGrid(GridWidth,GridHeight);
		//launch kernel function
		kernel_divideMFMaps<<<DimGrid,DimBlock>>>(device_DoGMap1,device_DoGMap2,device_DoGMap3,ImgWidth,param.PatchSize,Max_X,Max_Y,offset*i,PatchesWidth,device_PatchesData,device_PatchesEnergy);
	}
	cudaDeviceSynchronize(); //synchronize
	err = cudaGetLastError();
    if (err != cudaSuccess) printfErrorInfo("Failed to launch kernel function: kernel_divideMFMaps at wrap_divideReferMFMaps",cudaGetErrorString(err));
	// copy data from GPU to CPU
	float* host_PatchesData = (float*)malloc(size_PatchesData);
	float* host_PatchesEnergy = (float*)malloc(PatchesWidth*size_float);
	err = cudaMemcpy(host_PatchesData,device_PatchesData,size_PatchesData,cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) printfErrorInfo("cudaMemcpy Error occur when copying device_PatchesData to host_PatchesData",cudaGetErrorString(err));
	err = cudaMemcpy(host_PatchesEnergy,device_PatchesEnergy,PatchesWidth*size_float,cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) printfErrorInfo("cudaMemcpy Error occur when copying device_PatchesEnergy to host_PatchesEnergy",cudaGetErrorString(err));

	float MaxEnergy  = *max_element(host_PatchesEnergy,host_PatchesEnergy+PatchesWidth);
	param.EnergyThreshold = MaxEnergy * 0.05;
	//filtering patches with low energy
	for(int i = 0;i<PatchesWidth;i++)
	{
		if(host_PatchesEnergy[i] > param.EnergyThreshold)
		{
			param.vReferLocation.push_back(i);	//keep patches that over threshold line
		}
	}
	param.ReferWidth = param.vReferLocation.size();
	param.pReferMFPatches = (float*)malloc(param.ReferWidth * FeatureDim * size_float);
	for(int i = 0;i<param.ReferWidth;i++)
	{
		for(int j = 0;j < FeatureDim;j++)
		{
			param.pReferMFPatches[j*param.ReferWidth+i] = host_PatchesData[j*PatchesWidth+param.vReferLocation[i]];
		}
	}
	//free GPU and CPU memory
	cudaFree(device_DoGMap1);
	cudaFree(device_DoGMap2);
	cudaFree(device_DoGMap3);
	cudaFree(device_PatchesData);
	cudaFree(device_PatchesEnergy);
	free(host_PatchesData);
	free(host_PatchesEnergy);
}

//************************************
// Abstract:  launch CUDA kernel function to divide query feature maps(MF maps) into little patches
// FullName:  wrap_divideQueryMFMaps
// Access:    public 
// Returns:   void
// Qualifier:
// Parameter: vector<Mat * > & vQueryMFMaps
// Parameter: Parameters & param
//************************************
void wrap_divideQueryMFMaps(vector<Mat*> & vQueryMFMaps,Parameters & param)
{
	int size_float = sizeof(float);
	int Max_X = (vQueryMFMaps[0]->cols - param.PatchSize) / param.SlidingLength + 1;	//max numbers of patches in horizontal direction
	int Max_Y = (vQueryMFMaps[0]->rows - param.PatchSize) / param.SlidingLength + 1;	//max numbers of patches in vertical direction
	int PatchesWidth = Max_X * Max_Y;	//total patches
	int GridWidth,GridHeight;		    //grid dimension 
	int FeatureDim = param.PatchSize * param.PatchSize * 3;
	size_t size_PatchesData = PatchesWidth * FeatureDim * size_float;
	size_t size_DoGMap = vQueryMFMaps[0]->rows * vQueryMFMaps[0]->cols * size_float;
	dim3 DimBlock(BLOCK_SIZE,BLOCK_SIZE);

	float* device_DoGMap1 = NULL;
	float* device_DoGMap2 = NULL;
	float* device_DoGMap3 = NULL;
	float* device_PatchesData = NULL;
	float* device_PatchesEnergy = NULL;
	//allocate GPU global memory
	cudaError_t err = cudaSuccess;
	err = cudaMalloc((void **)&device_DoGMap1,size_DoGMap);
	if (err != cudaSuccess) printfErrorInfo("cudaMalloc Error at [device_DoGMap1]",cudaGetErrorString(err));  	

	err = cudaMalloc((void **)&device_DoGMap2,size_DoGMap);
	if (err != cudaSuccess) printfErrorInfo("cudaMalloc Error at [device_DoGMap2]",cudaGetErrorString(err));

	err = cudaMalloc((void **)&device_DoGMap3,size_DoGMap);
	if (err != cudaSuccess) printfErrorInfo("cudaMalloc Error at [device_DoGMap3]",cudaGetErrorString(err));

	err = cudaMalloc((void **)&device_PatchesData,size_PatchesData);
	if (err != cudaSuccess) printfErrorInfo("cudaMalloc Error at [device_PatchesData]",cudaGetErrorString(err));

	err = cudaMalloc((void **)&device_PatchesEnergy,PatchesWidth*size_float);
	if (err != cudaSuccess) printfErrorInfo("cudaMalloc Error at [device_PatchesEnergy]",cudaGetErrorString(err));

	//copy data from CPU to GPU
	err = cudaMemcpy(device_DoGMap1,(float*)vQueryMFMaps[0]->data,size_DoGMap,cudaMemcpyHostToDevice);
	if (err != cudaSuccess) printfErrorInfo("cudaMemcpy Error occur when copying vQueryMFMaps[0]->data to device_DoG1",cudaGetErrorString(err));
	err = cudaMemcpy(device_DoGMap2,(float*)vQueryMFMaps[1]->data,size_DoGMap,cudaMemcpyHostToDevice);
	if (err != cudaSuccess) printfErrorInfo("cudaMemcpy Error occur when copying vQueryMFMaps[1]->data to device_DoG1",cudaGetErrorString(err));
	err = cudaMemcpy(device_DoGMap3,(float*)vQueryMFMaps[2]->data,size_DoGMap,cudaMemcpyHostToDevice);
	if (err != cudaSuccess) printfErrorInfo("cudaMemcpy Error occur when copying vQueryMFMaps[2]->data to device_DoG1",cudaGetErrorString(err));
	
	int ImgWidth = vQueryMFMaps[0]->cols;
	int ImgHeight = vQueryMFMaps[0]->rows;
	GridWidth = (Max_X + BLOCK_SIZE - 1) / BLOCK_SIZE;
	GridHeight = (Max_Y + BLOCK_SIZE - 1)/ BLOCK_SIZE;
	dim3 DimGrid(GridWidth,GridHeight);
	//launch kernel function
	kernel_divideMFMaps<<<DimGrid,DimBlock>>>(device_DoGMap1,device_DoGMap2,device_DoGMap3,ImgWidth,param.PatchSize,Max_X,Max_Y,0,PatchesWidth,device_PatchesData,device_PatchesEnergy);
	cudaDeviceSynchronize(); //synchronize
	err = cudaGetLastError();
    if (err != cudaSuccess) printfErrorInfo("Failed to launch kernel function: kernel_divideMFMaps at wrap_divideQueryMFMaps",cudaGetErrorString(err));
	// copy data from GPU to CPU
	float* host_PatchesData = (float*)malloc(size_PatchesData);
	float* host_PatchesEnergy = (float*)malloc(PatchesWidth*size_float);
	err = cudaMemcpy(host_PatchesData,device_PatchesData,size_PatchesData,cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) printfErrorInfo("cudaMemcpy Error occur when copying device_PatchesData to host_PatchesData",cudaGetErrorString(err));
	err = cudaMemcpy(host_PatchesEnergy,device_PatchesEnergy,PatchesWidth*size_float,cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) printfErrorInfo("cudaMemcpy Error occur when copying device_PatchesData to host_PatchesData",cudaGetErrorString(err));

	//filtering patches with low energy
	vector<int> vQueryLocation;
	for(int i = 0;i<PatchesWidth;i++)
	{
		if(host_PatchesEnergy[i] > param.EnergyThreshold)
		{
			vQueryLocation.push_back(i);	//keep patches that over threshold line
		}
	}
	param.QueryWidth = vQueryLocation.size();
	param.pQueryMFPatches = (float*)malloc(param.QueryWidth * FeatureDim * size_float);
	param.pQueryLocation = (int*)malloc(param.QueryWidth * 2 *sizeof(int));

	for(int i = 0;i<param.QueryWidth;i++)
	{
		for(int j = 0;j < FeatureDim;j++)
		{
			param.pQueryMFPatches[j*param.QueryWidth+i] = host_PatchesData[j*PatchesWidth+vQueryLocation[i]];
		}
		int col = vQueryLocation[i] % Max_X;
		int row = vQueryLocation[i] / Max_X;
		param.pQueryLocation[i*2] = row;
		param.pQueryLocation[i*2+1] = col;
	}
	//free GPU and CPU memory
	cudaFree(device_DoGMap1);
	cudaFree(device_DoGMap2);
	cudaFree(device_DoGMap3);
	cudaFree(device_PatchesData);
	cudaFree(device_PatchesEnergy);
	free(host_PatchesData);
	free(host_PatchesEnergy);
	vector<int>().swap(vQueryLocation);
}

//************************************
// Abstract:  launch CUDA kernel function to divide high frequency details maps(HF maps) into patches
// FullName:  wrap_divideHFMaps
// Access:    public 
// Returns:   void
// Qualifier:
// Parameter: vector<Mat * > & vHFMaps
// Parameter: Parameters & param
//************************************
void wrap_divideHFMaps(vector<Mat* >& vHFMaps,Parameters & param)
{
	int size_float = sizeof(float);
	int Max_X = (vHFMaps[0]->cols - param.PatchSize) / param.SlidingLength + 1;	//max numbers of patches in horizontal direction
	int Max_Y = (vHFMaps[0]->rows - param.PatchSize) / param.SlidingLength + 1;	//max numbers of patches in vertical direction
	int offset = Max_X * Max_Y;
	int PatchesWidth = 4 * offset;	//total patches
	int GridWidth,GridHeight;		//grid dimension 
	int FeatureDim = param.PatchSize * param.PatchSize;
	size_t size_PatchesData = PatchesWidth * FeatureDim * size_float;
	size_t size_HFMap = vHFMaps[0]->rows * vHFMaps[0]->cols * size_float;
	dim3 DimBlock(BLOCK_SIZE,BLOCK_SIZE);

	float* device_HFMap = NULL;
	float* device_PatchesData = NULL;
	//allocate GPU global memory 
	cudaError_t err = cudaSuccess;
	err = cudaMalloc((void **)&device_HFMap,size_HFMap);	
	if (err != cudaSuccess) printfErrorInfo("cudaMalloc Error at [device_HFMap]",cudaGetErrorString(err)); 
	err = cudaMalloc((void **)&device_PatchesData,size_PatchesData);
	if (err != cudaSuccess) printfErrorInfo("cudaMalloc Error at [device_PatchesData]",cudaGetErrorString(err)); 

	for(int i = 0;i < 4;i++)
	{
		err = cudaMemcpy(device_HFMap,(float*)vHFMaps[i]->data,size_HFMap,cudaMemcpyHostToDevice);
		if (err != cudaSuccess) printfErrorInfo("cudaMemcpy Error occur when copying vHFMaps[i]->data->data to device_HFMap",cudaGetErrorString(err));
		
		int ImgWidth = vHFMaps[i]->cols;
		int ImgHeight = vHFMaps[i]->rows;
		int Max_X = (ImgWidth - param.PatchSize) / param.SlidingLength + 1;
		int Max_Y = (ImgHeight - param.PatchSize)/ param.SlidingLength + 1;
		GridWidth = (Max_X + BLOCK_SIZE - 1) / BLOCK_SIZE;
		GridHeight = (Max_Y + BLOCK_SIZE - 1)/ BLOCK_SIZE;
		dim3 DimGrid(GridWidth,GridHeight);
		kernel_divideHFMaps<<<DimGrid,DimBlock>>>(device_HFMap,ImgWidth,param.PatchSize,Max_X,Max_Y,offset*i,PatchesWidth,device_PatchesData);
	}
	cudaDeviceSynchronize(); //synchronize
	err = cudaGetLastError();
    if (err != cudaSuccess) printfErrorInfo("Failed to launch kernel function: kernel_divideHFMaps",cudaGetErrorString(err));
	
	float* host_PatchesData = (float*)malloc(size_PatchesData);
	err = cudaMemcpy(host_PatchesData,device_PatchesData,size_PatchesData,cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) printfErrorInfo("cudaMemcpy Error occur when copying device_PatchesData to host_PatchesData at wrap_divideHFMaps",cudaGetErrorString(err));
	//filtering HF patches according to MF patches
	param.pReferHFPatches = (float*)malloc(param.ReferWidth * FeatureDim * size_float);
	for(int i = 0; i < param.ReferWidth ;i++)
	{
		for(int j = 0;j < FeatureDim;j++)
		{
			param.pReferHFPatches[j*param.ReferWidth+i] = host_PatchesData[j*PatchesWidth+param.vReferLocation[i]];
		}
	}
	//free GPU and CPU memory
	cudaFree(device_HFMap);
	cudaFree(device_PatchesData);
	free(host_PatchesData);
}


