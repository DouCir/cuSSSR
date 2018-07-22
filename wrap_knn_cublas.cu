#include "kernel_knn_cublas.cu"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

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
void printErrorMessage(cudaError_t error, int memorySize)
{
    printf("==================================================\n");
    printf("MEMORY ALLOCATION ERROR  : %s\n", cudaGetErrorString(error));
    printf("Whished allocated memory : %d\n", memorySize);
    printf("==================================================\n");
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
void wrap_knn_cublas(float* ref_host, int ref_width, float* query_host, int query_width, int height, int k, float* dist_host, int* ind_host)
{
    
    unsigned int size_of_float = sizeof(float);
    unsigned int size_of_int   = sizeof(int);
    
    // Variables
    float        *query_dev;
    float        *ref_dev;
    float        *dist_dev;
    float        *query_norm;
    float        *ref_norm;
    int          *ind_dev;
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
    
    // CUDA Initialisation
    cuInit(0);
    
    // Check free memory using driver API ; only (MAX_PART_OF_FREE_MEMORY_USED*100)% of memory will be used
    CUcontext cuContext;
    CUdevice  cuDevice=0;
    cuCtxCreate(&cuContext, 0, cuDevice);
    cuMemGetInfo(&memory_free, &memory_total);
    cuCtxDetach (cuContext);

    printf("total memory: %u,free memory: %u\n",memory_total,memory_free);

    // Determine maximum number of query that can be treated
    max_nb_query_traited = ( memory_free * MAX_PART_OF_FREE_MEMORY_USED - size_of_float * ref_width*(height+1) ) / ( size_of_float * (height + ref_width + 1) + size_of_int * k);
    max_nb_query_traited = min( (size_t)query_width, (max_nb_query_traited / 16) * 16 );
    
	printf("max_nb_query_traited:%u,query_width:%d\n",max_nb_query_traited,query_width);
	printf("refer_width:%d\n",ref_width);

    // Allocation of global memory for query points and for distances
    result = cudaMallocPitch( (void **) &query_dev, &query_pitch_in_bytes, max_nb_query_traited * size_of_float, height + ref_width + 1);
    if (result)
	{
        printErrorMessage(result, max_nb_query_traited*size_of_float*(height+ref_width));
		printf("Error: 1\n");
        return;
    }
    query_pitch = query_pitch_in_bytes/size_of_float;
    query_norm  = query_dev  + height * query_pitch;
    dist_dev    = query_norm + query_pitch;
    
    // Allocation of global memory for reference points and ||query||
    result = cudaMallocPitch((void **) &ref_dev, &ref_pitch_in_bytes, ref_width * size_of_float, height+1);
    if (result){
        printErrorMessage(result, ref_width * size_of_float * ( height+1 ));
        cudaFree(query_dev);
		printf("Error 2\n");
        return;
    }
    ref_pitch = ref_pitch_in_bytes / size_of_float;
    ref_norm  = ref_dev + height * ref_pitch;
	
    // Allocation of global memory for indexes	
    result = cudaMallocPitch( (void **) &ind_dev, &ind_pitch_in_bytes, max_nb_query_traited * size_of_int, k);
    if (result){
        printErrorMessage(result, max_nb_query_traited*size_of_int*k);
        cudaFree(ref_dev);
        cudaFree(query_dev);
		printf("Error 3\n");
        return;
    }
    ind_pitch = ind_pitch_in_bytes/size_of_int;
    
    // Memory copy of ref_host in ref_dev
    result = cudaMemcpy2D(ref_dev, ref_pitch_in_bytes, ref_host, ref_width*size_of_float, ref_width*size_of_float, height, cudaMemcpyHostToDevice);
    
    // Computation of reference square norm
    dim3 ref_grid(ref_width/256, 1, 1);
    dim3 ref_thread(256, 1, 1);
    if (ref_width%256 != 0) ref_grid.x += 1;
    cuComputeNorm<<<ref_grid,ref_thread>>>(ref_dev, ref_width, ref_pitch, height, ref_norm);
    
    // Split queries to fit in GPU memory
    for (int i=0; i<query_width; i+=max_nb_query_traited){
        
		// Number of query points considered
        actual_nb_query_width = min( max_nb_query_traited, (size_t)(query_width-i) );
        
        // Copy of part of query actually being treated
        cudaMemcpy2D(query_dev, query_pitch_in_bytes, &query_host[i], query_width*size_of_float, actual_nb_query_width*size_of_float, height, cudaMemcpyHostToDevice);
        
        // Computation of Q square norm
        dim3 query_grid_1(actual_nb_query_width/256, 1, 1);
        dim3 query_thread_1(256, 1, 1);
        if (actual_nb_query_width%256 != 0) query_grid_1.x += 1;
        cuComputeNorm<<<query_grid_1,query_thread_1>>>(query_dev, actual_nb_query_width, query_pitch, height, query_norm);
        
        // Computation of Q*transpose(R)
        cublasSgemm('n', 't', (int)query_pitch, (int)ref_pitch, height, (float)-2.0, query_dev, query_pitch, ref_dev, ref_pitch, (float)0.0, dist_dev, query_pitch);
        
        // Add R norm to distances
        dim3 query_grid_2(actual_nb_query_width/16, ref_width/16, 1);
        dim3 query_thread_2(16, 16, 1);
        if (actual_nb_query_width%16 != 0) query_grid_2.x += 1;
        if (ref_width%16 != 0) query_grid_2.y += 1;
        cuAddRNorm<<<query_grid_2,query_thread_2>>>(dist_dev, actual_nb_query_width, query_pitch, ref_width, ref_norm);
        
        // Sort each column
        cuInsertionSort<<<query_grid_1,query_thread_1>>>(dist_dev, query_pitch, ind_dev, ind_pitch, actual_nb_query_width, ref_width, k);
        
        // Add Q norm and compute Sqrt ONLY ON ROW K-1
        cuAddQNormAndSqrt<<<query_grid_2,query_thread_2>>>( dist_dev, actual_nb_query_width, query_pitch, query_norm, k);
        
        // Memory copy of output from device to host
        cudaMemcpy2D(&dist_host[i], query_width*size_of_float, dist_dev, query_pitch_in_bytes, actual_nb_query_width*size_of_float, k, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(&ind_host[i],  query_width*size_of_int,   ind_dev,  ind_pitch_in_bytes,   actual_nb_query_width*size_of_int,   k, cudaMemcpyDeviceToHost);
    }
    
    // Free memory
    cudaFree(ind_dev);
    cudaFree(ref_dev);
    cudaFree(query_dev);
}


