#include <stdio.h>

int main( void ) {

  cudaDeviceProp  prop;
  int count;
  cudaGetDeviceCount(&count);

  for (int i=0; i< count; i++) {

    cudaGetDeviceProperties(&prop, i);

    printf( "   --- General Information for device %d ---\n", i );
    printf( "Device %d: %s\n", i, prop.name );
    printf( "CUDA capability Major.Minor version:  %d.%d\n", prop.major, prop.minor );

    printf( "Total global mem:                     %.0f MBytes  (%ld bytes)\n", prop.totalGlobalMem/1048576.0f, prop.totalGlobalMem );

    printf( "GPU Max Clock rate:                   %.0f MHz (%0.2f GHz)\n", prop.clockRate*1e-3f,prop.clockRate*1e-6f );
    printf("Memory Clock rate:                    %.0f Mhz\n", prop.memoryClockRate * 1e-3f);

    printf("Memory Bus Width:                     %d-bit\n",   prop.memoryBusWidth);




    printf( "Total constant memory:                %ld bytes\n", prop.totalConstMem );
    printf( "Shared memory per block:              %ld bytes\n", prop.sharedMemPerBlock );
    printf( "Registers per block:                  %d\n", prop.regsPerBlock );
    printf( "Warp size:                            %d\n", prop.warpSize );
    printf( "Max memory pitch:                     %ld bytes\n", prop.memPitch );
    printf( "Texture Alignment:                    %ld bytes\n", prop.textureAlignment );

    printf( "Multiprocessor count:                 %d\n",
                            prop.multiProcessorCount );

    printf( "Max threads per block:                %d\n",
        
                            prop.maxThreadsPerBlock );

    printf( "Max thread block dimensions (x,y,z):  (%d, %d, %d)\n",
        
                            prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                            
                                                prop.maxThreadsDim[2] );

    printf( "Max grid dimensions (x,y,z):          (%d, %d, %d)\n",
        
                            prop.maxGridSize[0], prop.maxGridSize[1],
                            
                                                prop.maxGridSize[2] );

    printf( "Concurrent copy and kernel execution: " );

    if (prop.deviceOverlap)
        printf( "Enabled\n" );
    else
        printf( "Disabled\n");

    printf( "Run time limit on kernels :           " );

    if (prop.kernelExecTimeoutEnabled)
        printf( "Enabled\n" );
    else
        printf( "Disabled\n" );

    printf( "\n" );
  }

}
