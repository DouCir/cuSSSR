/**
  *
  * Date         03/07/2009
  * ====
  *
  * Authors      Vincent Garcia
  * =======      Eric    Debreuve
  *              Michel  Barlaud
  *
  * Description  Given a reference point set and a query point set, the program returns
  * ===========  firts the distance between each query point and its k nearest neighbors in
  *              the reference point set, and second the indexes of these k nearest neighbors.
  *              The computation is performed using the API NVIDIA CUDA.
  *
  * Paper        Fast k nearest neighbor search using GPU
  * =====
  *
  * BibTeX       @INPROCEEDINGS{2008_garcia_cvgpu,
  * ======         author = {V. Garcia and E. Debreuve and M. Barlaud},
  *                title = {Fast k nearest neighbor search using GPU},
  *                booktitle = {CVPR Workshop on Computer Vision on GPU},
  *                year = {2008},
  *                address = {Anchorage, Alaska, USA},
  *                month = {June}
  *              }
  *
  */

//standard library
#include <stdio.h>
#include <math.h>
#include <time.h>
//CUDA runtime library
#include <cuda.h>
#include <cuda_runtime.h>
//kernel function
#include "kernel_knn_cuda.cu"

// Constants used by the program
#define MAX_PITCH_VALUE_IN_BYTES       262144
#define MAX_TEXTURE_WIDTH_IN_BYTES     65536
#define MAX_TEXTURE_HEIGHT_IN_BYTES    32768
#define MAX_PART_OF_FREE_MEMORY_USED   0.9
#define BLOCK_DIM                      16

//-----------------------------------------------------------------------------------------------//
//                                   K-th NEAREST NEIGHBORS                                      //
//-----------------------------------------------------------------------------------------------//



/**
  * Prints the error message return during the memory allocation.
  *
  * @param error        error value return by the memory allocation function
  * @param memorySize   size of memory tried to be allocated
  */
void printErrorMessage(cudaError_t error, int memorySize){
    printf("==================================================\n");
    printf("MEMORY ALLOCATION ERROR  : %s\n", cudaGetErrorString(error));
    printf("Whished allocated memory : %d\n", memorySize);
    printf("==================================================\n");
#if MATLAB_CODE == 1
    mexErrMsgTxt("CUDA ERROR DURING MEMORY ALLOCATION");
#endif
}



/**
  * K nearest neighbor algorithm
  * - Initialize CUDA
  * - Allocate device memory
  * - Copy point sets (reference and query points) from host to device memory
  * - Compute the distances + indexes to the k nearest neighbors for each query point
  * - Copy distances from device to host memory
  *
  * @param ref_host      reference points ; pointer to linear matrix
  * @param ref_width     number of reference points ; width of the matrix
  * @param query_host    query points ; pointer to linear matrix
  * @param query_width   number of query points ; width of the matrix
  * @param height        dimension of points ; height of the matrices
  * @param k             number of neighbor to consider
  * @param dist_host     distances to k nearest neighbors ; pointer to linear matrix
  * @param dist_host     indexes of the k nearest neighbors ; pointer to linear matrix
  *
  */
void wrap_KNN(float* ref_host, int ref_width, float* query_host, int query_width, int height, int k, float* dist_host, int* ind_host){
    
    unsigned int size_of_float = sizeof(float);
    unsigned int size_of_int   = sizeof(int);
    
    // Variables
    float        *query_dev;
    float        *ref_dev;
    float        *dist_dev;
    int          *ind_dev;
    cudaArray    *ref_array;
    cudaError_t  result;
    size_t       query_pitch;
    size_t	     query_pitch_in_bytes;
    size_t       ref_pitch;
    size_t       ref_pitch_in_bytes;
    size_t       ind_pitch;
    size_t       ind_pitch_in_bytes;
    size_t       max_nb_query_traited;
    size_t       actual_nb_query_width;
    size_t       memory_total;
    size_t       memory_free;
    
	
	  // Check if we can use texture memory for reference points
    unsigned int use_texture = ( ref_width*size_of_float<=MAX_TEXTURE_WIDTH_IN_BYTES && height*size_of_float<=MAX_TEXTURE_HEIGHT_IN_BYTES );
    
    // CUDA Initialisation
    cuInit(0);
    
    // Check free memory using driver API ; only (MAX_PART_OF_FREE_MEMORY_USED*100)% of memory will be used
    CUcontext cuContext;
    CUdevice  cuDevice=0;
    cuCtxCreate(&cuContext, 0, cuDevice);
    cuMemGetInfo(&memory_free, &memory_total);
    cuCtxDetach (cuContext);
    
    // Determine maximum number of query that can be treated
    max_nb_query_traited = ( memory_free * MAX_PART_OF_FREE_MEMORY_USED - size_of_float * ref_width*height ) / ( size_of_float * (height + ref_width) + size_of_int * k);
    max_nb_query_traited = min( (size_t)query_width, (max_nb_query_traited / 16) * 16 );
    
    // Allocation of global memory for query points and for distances
    result = cudaMallocPitch( (void **) &query_dev, &query_pitch_in_bytes, max_nb_query_traited * size_of_float, height + ref_width);
    if (result){
        printErrorMessage(result, max_nb_query_traited*size_of_float*(height+ref_width));
        return;
    }
    query_pitch = query_pitch_in_bytes/size_of_float;
    dist_dev    = query_dev + height * query_pitch;
	
    // Allocation of global memory for indexes	
    result = cudaMallocPitch( (void **) &ind_dev, &ind_pitch_in_bytes, max_nb_query_traited * size_of_int, k);
	if (result){
        cudaFree(query_dev);
        printErrorMessage(result, max_nb_query_traited*size_of_int*k);
        return;
    }
    ind_pitch = ind_pitch_in_bytes/size_of_int;
    
    // Allocation of memory (global or texture) for reference points
    if (use_texture){
	
        // Allocation of texture memory
        cudaChannelFormatDesc channelDescA = cudaCreateChannelDesc<float>();
        result = cudaMallocArray( &ref_array, &channelDescA, ref_width, height );
        if (result){
            printErrorMessage(result, ref_width*height*size_of_float);
            cudaFree(ind_dev);
            cudaFree(query_dev);
            return;
        }
        cudaMemcpyToArray( ref_array, 0, 0, ref_host, ref_width * height * size_of_float, cudaMemcpyHostToDevice );
        
        // Set texture parameters and bind texture to array
        texA.addressMode[0] = cudaAddressModeClamp;
        texA.addressMode[1] = cudaAddressModeClamp;
        texA.filterMode     = cudaFilterModePoint;
        texA.normalized     = 0;
        cudaBindTextureToArray(texA, ref_array);
		
    }
    else{
	
		// Allocation of global memory
        result = cudaMallocPitch( (void **) &ref_dev, &ref_pitch_in_bytes, ref_width * size_of_float, height);
        if (result){
            printErrorMessage(result,  ref_width*size_of_float*height);
            cudaFree(ind_dev);
            cudaFree(query_dev);
            return;
        }
        ref_pitch = ref_pitch_in_bytes/size_of_float;
        cudaMemcpy2D(ref_dev, ref_pitch_in_bytes, ref_host, ref_width*size_of_float,  ref_width*size_of_float, height, cudaMemcpyHostToDevice);
    }
    
    // Split queries to fit in GPU memory
    for (int i=0; i<query_width; i+=max_nb_query_traited){
        
		// Number of query points considered
        actual_nb_query_width = min( max_nb_query_traited, (size_t)(query_width-i));
        
        // Copy of part of query actually being treated
        cudaMemcpy2D(query_dev, query_pitch_in_bytes, &query_host[i], query_width*size_of_float, actual_nb_query_width*size_of_float, height, cudaMemcpyHostToDevice);
        
        // Grids ans threads
        dim3 g_16x16(actual_nb_query_width/16, ref_width/16, 1);
        dim3 t_16x16(16, 16, 1);
        if (actual_nb_query_width%16 != 0) g_16x16.x += 1;
        if (ref_width  %16 != 0) g_16x16.y += 1;
        //
        dim3 g_256x1(actual_nb_query_width/256, 1, 1);
        dim3 t_256x1(256, 1, 1);
        if (actual_nb_query_width%256 != 0) g_256x1.x += 1;
    		//
        dim3 g_k_16x16(actual_nb_query_width/16, k/16, 1);
        dim3 t_k_16x16(16, 16, 1);
        if (actual_nb_query_width%16 != 0) g_k_16x16.x += 1;
        if (k  %16 != 0) g_k_16x16.y += 1;
        
        // Kernel 1: Compute all the distances
        if (use_texture)
            cuComputeDistanceTexture<<<g_16x16,t_16x16>>>(ref_width, query_dev, actual_nb_query_width, query_pitch, height, dist_dev);
        else
            cuComputeDistanceGlobal<<<g_16x16,t_16x16>>>(ref_dev, ref_width, ref_pitch, query_dev, actual_nb_query_width, query_pitch, height, dist_dev);
            
        // Kernel 2: Sort each column
        cuInsertionSort<<<g_256x1,t_256x1>>>(dist_dev, query_pitch, ind_dev, ind_pitch, actual_nb_query_width, ref_width, k);
        
        // Kernel 3: Compute square root of k first elements
        cuParallelSqrt<<<g_k_16x16,t_k_16x16>>>(dist_dev, query_width, query_pitch, k);
        
        // Memory copy of output from device to host
		cudaMemcpy2D(&dist_host[i], query_width*size_of_float, dist_dev, query_pitch_in_bytes, actual_nb_query_width*size_of_float, k, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(&ind_host[i],  query_width*size_of_int,   ind_dev,  ind_pitch_in_bytes,   actual_nb_query_width*size_of_int,   k, cudaMemcpyDeviceToHost);
    }
    
    // Free memory
    if (use_texture)
        cudaFreeArray(ref_array);
    else
        cudaFree(ref_dev);
    cudaFree(ind_dev);
    cudaFree(query_dev);

	//cudaDeviceSynchronize();
	//cudaError_t err = cudaSuccess;
	//err = cudaGetLastError();
	//if (err != cudaSuccess) printf("Error occur in wrap_KNN\n");
}

