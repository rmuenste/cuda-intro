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

    cudaStream_t    stream0, stream1;
    int *host_a, *host_b, *host_c;
    int *dev_a0, *dev_b0, *dev_c0;
    int *dev_a1, *dev_b1, *dev_c1;

    // start the timers
    checkCudaErrors( cudaEventCreate( &start ) );
    checkCudaErrors( cudaEventCreate( &stop ) );

    // initialize the streams
    checkCudaErrors( cudaStreamCreate( &stream0 ) );
    checkCudaErrors( cudaStreamCreate( &stream1 ) );

    // allocate the memory on the GPU
    checkCudaErrors( cudaMalloc( (void**)&dev_a0,
                              N * sizeof(int) ) );
    checkCudaErrors( cudaMalloc( (void**)&dev_b0,
                              N * sizeof(int) ) );
    checkCudaErrors( cudaMalloc( (void**)&dev_c0,
                              N * sizeof(int) ) );
    checkCudaErrors( cudaMalloc( (void**)&dev_a1,
                              N * sizeof(int) ) );
    checkCudaErrors( cudaMalloc( (void**)&dev_b1,
                              N * sizeof(int) ) );
    checkCudaErrors( cudaMalloc( (void**)&dev_c1,
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
    for (int i=0; i<FULL_DATA_SIZE; i+= N*2) {
        // enqueue copies of a in stream0 and stream1
        checkCudaErrors( cudaMemcpyAsync( dev_a0, host_a+i,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream0 ) );
        checkCudaErrors( cudaMemcpyAsync( dev_a1, host_a+i+N,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream1 ) );
        // enqueue copies of b in stream0 and stream1
        checkCudaErrors( cudaMemcpyAsync( dev_b0, host_b+i,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream0 ) );
        checkCudaErrors( cudaMemcpyAsync( dev_b1, host_b+i+N,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream1 ) );

        // enqueue kernels in stream0 and stream1   
        kernel<<<N/256,256,0,stream0>>>( dev_a0, dev_b0, dev_c0 );
        kernel<<<N/256,256,0,stream1>>>( dev_a1, dev_b1, dev_c1 );

        // enqueue copies of c from device to locked memory
        checkCudaErrors( cudaMemcpyAsync( host_c+i, dev_c0,
                                       N * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       stream0 ) );
        checkCudaErrors( cudaMemcpyAsync( host_c+i+N, dev_c1,
                                       N * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       stream1 ) );
    }
    checkCudaErrors( cudaStreamSynchronize( stream0 ) );
    checkCudaErrors( cudaStreamSynchronize( stream1 ) );

    checkCudaErrors( cudaEventRecord( stop, 0 ) );

    checkCudaErrors( cudaEventSynchronize( stop ) );
    checkCudaErrors( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
    printf( "Time taken:  %3.1f ms\n", elapsedTime );

    // cleanup the streams and memory
    checkCudaErrors( cudaFreeHost( host_a ) );
    checkCudaErrors( cudaFreeHost( host_b ) );
    checkCudaErrors( cudaFreeHost( host_c ) );
    checkCudaErrors( cudaFree( dev_a0 ) );
    checkCudaErrors( cudaFree( dev_b0 ) );
    checkCudaErrors( cudaFree( dev_c0 ) );
    checkCudaErrors( cudaFree( dev_a1 ) );
    checkCudaErrors( cudaFree( dev_b1 ) );
    checkCudaErrors( cudaFree( dev_c1 ) );
    checkCudaErrors( cudaStreamDestroy( stream0 ) );
    checkCudaErrors( cudaStreamDestroy( stream1 ) );

    return 0;
}

