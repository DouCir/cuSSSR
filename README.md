# cuSSSR
The code of paper "A Fast Single-Image Super-Resolution Method Implemented With CUDA"

## Constant 

### src--the source code

    main.cpp:the entry point
    cuSSSR.h/cuSSSR.cpp: declaration and implementation of functions which run on CPU
    kernel_divideFeatureMaps.cu: the CUDA kernel function sets which divide image feature maps into little patches.
    kernel_estimateHF.cu: the CUDA kernel funcion sets which estimate high frequency(HF) details of high resolution image.
    kernel_knn_cuda: the implementation of KNN(K-Nearest Neighbor)algorithm with base CUDA library by Vincent Garcia etc.
    kernel_knn_cublas:the implementation of KNN(K-Nearest Neighbor)algorithm with cublas library by Vincent Garcia etc.
    wrap_knn.cu: the C API of KNN CUDA version.
    wrap_knn_cublas: the C API of KNN cublas version.
    wrap_estimateHF.cu: the C API of estimating HF details of high resolut on image.
    warp_devideFeatureMaps.cu:the C API of dividing image feature maps into into little patches.
    
### images--test images
    
    some test images which cover Set5,Set14,BSDS100,Urban100. Groud truth images and corresponding low resolution(LR) are given.
    
