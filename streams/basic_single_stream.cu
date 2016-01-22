/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include <stdio.h>
#include <cuda_runtime.h>

#define N   (1024*1024)
#define FULL_DATA_SIZE   (N*20)

#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

void __checkCudaErrors(cudaError_t err, const char *file, const int line)
{

  if (cudaSuccess != err)
  {
    fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n", err, cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }

}

#endif


__global__ void kernel( int *a, int *b, int *c ) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float   as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float   bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}
        

int main( void ) {
    cudaDeviceProp  prop;
    int whichDevice;
    checkCudaErrors( cudaGetDevice( &whichDevice ) );
    checkCudaErrors( cudaGetDeviceProperties( &prop, whichDevice ) );
    if (!prop.deviceOverlap) {
        printf( "Device will not handle overlaps, so no speed up from streams\n" );
        return 0;
    }

    cudaEvent_t     start, stop;
    float           elapsedTime;

    cudaStream_t    stream;
    int *host_a, *host_b, *host_c;
    int *dev_a, *dev_b, *dev_c;

    // start the timers
    checkCudaErrors( cudaEventCreate( &start ) );
    checkCudaErrors( cudaEventCreate( &stop ) );

    // initialize the stream
    checkCudaErrors( cudaStreamCreate( &stream ) );

    // allocate the memory on the GPU
    checkCudaErrors( cudaMalloc( (void**)&dev_a,
                              N * sizeof(int) ) );
    checkCudaErrors( cudaMalloc( (void**)&dev_b,
                              N * sizeof(int) ) );
    checkCudaErrors( cudaMalloc( (void**)&dev_c,
                              N * sizeof(int) ) );

    // allocate host locked memory, used to stream
    checkCudaErrors( cudaHostAlloc( (void**)&host_a,
                              FULL_DATA_SIZE * sizeof(int),
                              cudaHostAllocDefault ) );
    checkCudaErrors( cudaHostAlloc( (void**)&host_b,
                              FULL_DATA_SIZE * sizeof(int),
                              cudaHostAllocDefault ) );
    checkCudaErrors( cudaHostAlloc( (void**)&host_c,
                              FULL_DATA_SIZE * sizeof(int),
                              cudaHostAllocDefault ) );

    for (int i=0; i<FULL_DATA_SIZE; i++) {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    checkCudaErrors( cudaEventRecord( start, 0 ) );
    // now loop over full data, in bite-sized chunks
    for (int i=0; i<FULL_DATA_SIZE; i+= N) {
        // copy the locked memory to the device, async
        checkCudaErrors( cudaMemcpyAsync( dev_a, host_a+i,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream ) );
        checkCudaErrors( cudaMemcpyAsync( dev_b, host_b+i,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream ) );

        kernel<<<N/256,256,0,stream>>>( dev_a, dev_b, dev_c );

        // copy the data from device to locked memory
        checkCudaErrors( cudaMemcpyAsync( host_c+i, dev_c,
                                       N * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       stream ) );

    }
    // copy result chunk from locked to full buffer
    checkCudaErrors( cudaStreamSynchronize( stream ) );

    checkCudaErrors( cudaEventRecord( stop, 0 ) );

    checkCudaErrors( cudaEventSynchronize( stop ) );
    checkCudaErrors( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
    printf( "Time taken:  %3.1f ms\n", elapsedTime );

    // cleanup the streams and memory
    checkCudaErrors( cudaFreeHost( host_a ) );
    checkCudaErrors( cudaFreeHost( host_b ) );
    checkCudaErrors( cudaFreeHost( host_c ) );
    checkCudaErrors( cudaFree( dev_a ) );
    checkCudaErrors( cudaFree( dev_b ) );
    checkCudaErrors( cudaFree( dev_c ) );
    checkCudaErrors( cudaStreamDestroy( stream ) );

    return 0;
}

