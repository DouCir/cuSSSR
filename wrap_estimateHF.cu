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
//CUDA runtime library
#include <cuda.h>
#include <cuda_runtime.h>
//OpenCV library
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
//kernel function
#include "kernel_estimateHF.cu"

#include "cuSSSR.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

void printfErrorInfo(const char* DisplayStr,const char* ErrString)
{
	printf("Error! %s! more details: %s \n",DisplayStr,ErrString);
	system("pause");
	return;
}

void wrap_estimateHR(const Mat & InputIntpHR,Mat & OutputHR,Parameters &param)
{
	int Width = InputIntpHR.cols;
	int Height = InputIntpHR.rows;
	//calculate how many memory (in bytes) will be used on GPU
	size_t size_ReferHF = param.ReferWidth*param.PatchSize*param.PatchSize*sizeof(float);
	size_t size_Dist = param.QueryWidth*param.K*sizeof(float);
	size_t size_Idx = param.QueryWidth*param.K*sizeof(int);
	size_t size_EstimatedHF = Width*Height*sizeof(float);
	size_t size_Overlap = Width*Height*sizeof(int);
	size_t size_Val = param.QueryWidth*param.PatchSize*param.PatchSize*param.K*sizeof(float);
	size_t size_Location = param.QueryWidth*2*sizeof(int);
	//allocate global memory on GPU
	cudaError_t err = cudaSuccess;
	float* device_ReferHF = NULL;
	err = cudaMalloc((void **)&device_ReferHF,size_ReferHF);
	if (err != cudaSuccess) printfErrorInfo("cudaMalloc Error at [device_ReferHF]",cudaGetErrorString(err));

	float* device_KNNDist = NULL;
	err = cudaMalloc((void **)&device_KNNDist,size_Dist);
	if (err != cudaSuccess) printfErrorInfo("cudaMalloc Error at [device_KNNDist]",cudaGetErrorString(err));

	int* device_KNNIndex = NULL;
	err= cudaMalloc((void **)&device_KNNIndex,size_Idx);
	if (err != cudaSuccess) printfErrorInfo("cudaMalloc Error at [device_KNNIndex]",cudaGetErrorString(err));

	float* device_EstimatedHF = NULL;
	err = cudaMalloc((void **)&device_EstimatedHF,size_EstimatedHF);
	if (err != cudaSuccess) printfErrorInfo("cudaMalloc Error at [device_EstimatedHF]",cudaGetErrorString(err));
	cudaMemset(device_EstimatedHF,0.0,size_EstimatedHF);

	int* device_OverlapArea = NULL;
	cudaMalloc((void **)&device_OverlapArea,size_Overlap);
	cudaMemset(device_OverlapArea,0,size_Overlap);
	if (err != cudaSuccess) printfErrorInfo("cudaMalloc Error at [device_OverlapArea]",cudaGetErrorString(err));

	float* device_Average = NULL;
	err = cudaMalloc((void **)&device_Average,size_EstimatedHF);
	if (err != cudaSuccess) printfErrorInfo("cudaMalloc Error at [device_Average]",cudaGetErrorString(err));

	int* device_Location = NULL;
	err = cudaMalloc((void **)&device_Location,size_Location);
	if (err != cudaSuccess) printfErrorInfo("cudaMalloc Error at [device_Location]",cudaGetErrorString(err)); 

	//copy data from host(CPU) to device(GPU)
	err = cudaMemcpy(device_ReferHF,param.pReferHFPatches,size_ReferHF,cudaMemcpyHostToDevice);
	if (err != cudaSuccess) printfErrorInfo("cudaMemcpy Error at copying param.pReferHFPatches to device_ReferHF",cudaGetErrorString(err));

	err = cudaMemcpy(device_KNNDist,param.pKNNDist,size_Dist,cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printfErrorInfo("cudaMemcpy Error at copying param.pKNNDist to device_KNNDist",cudaGetErrorString(err));
	
	err = cudaMemcpy(device_KNNIndex,param.pKNNIndex,size_Idx,cudaMemcpyHostToDevice);
	if (err != cudaSuccess) printfErrorInfo("cudaMemcpy Error at copying param.pKNNIndex to device_KNNIndex",cudaGetErrorString(err));

	err = cudaMemcpy(device_Location,param.pQueryLocation,size_Location,cudaMemcpyHostToDevice);
	if (err != cudaSuccess) printfErrorInfo("cudaMemcpy Error at copying param.pQueryLocation to device_Location",cudaGetErrorString(err));

	//temp variable
	float* Val = NULL;
	err = cudaMalloc((void **)&Val,size_Val);
	if (err != cudaSuccess) printfErrorInfo("cudaMalloc Error at [Val]",cudaGetErrorString(err)); 

	int threadsPerBlock = 256;
    int blocksPerGrid =(param.QueryWidth + threadsPerBlock - 1) / threadsPerBlock;
	//launch kernel function 1: rebuild HF details
	kernel_estimateHF<<<dim3(blocksPerGrid),dim3(threadsPerBlock)>>>(device_ReferHF,device_KNNDist,device_KNNIndex,device_Location,Val,param.PatchSize,param.K,Width,Height,param.ReferWidth,param.QueryWidth,device_EstimatedHF,device_OverlapArea);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
    if (err != cudaSuccess) printfErrorInfo("Failed to launch kernel function: kernel_estimateHF",cudaGetErrorString(err));

	int GridWidth = (Width+BLOCK_SIZE-1)/BLOCK_SIZE;
	int GridHeight = (Height+BLOCK_SIZE-1)/BLOCK_SIZE;
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 dimOverlap(GridWidth,GridHeight);
	//launch kernel function 2:Process Overlaps between patches
	kernel_processOverlapArea<<<dimOverlap,dimBlock>>>(device_EstimatedHF,device_OverlapArea,Width,Height,param.PatchSize,device_Average);
	err = cudaGetLastError();
    if (err != cudaSuccess) printfErrorInfo("Failed to launch kernel function: kernel_processOverlapArea",cudaGetErrorString(err));
	//copy HF image from device(GPU) to host(CPU)
	Mat HFData(InputIntpHR.size(),InputIntpHR.type());
	err = cudaMemcpy(HFData.data,(char*)device_Average,size_EstimatedHF,cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) printfErrorInfo("cudaMemcpy Error at copying device_Average to HFData.data",cudaGetErrorString(err));

	//step 14
	OutputHR = InputIntpHR + HFData;
	if (param.ZoomFactor < 0.0) OutputHR.convertTo(OutputHR,CV_8UC1);

	//free global memory allocated in GPU
	cudaFree(device_ReferHF);
	cudaFree(device_KNNDist);
	cudaFree(device_KNNIndex);
	cudaFree(device_EstimatedHF);
	cudaFree(device_OverlapArea);
	cudaFree(device_Average);
	cudaFree(Val);
}