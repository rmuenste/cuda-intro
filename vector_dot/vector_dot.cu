#include <stdio.h>
#include <cuda_runtime.h>

#define N (33 * 1024)

#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

void __checkCudaErrors(cudaError_t err, const char *file, const int line)
{

  if(cudaSuccess != err)
  {
    fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n", err, cudaGetErrorString(err), file, line); 
    exit(EXIT_FAILURE);
  }

}

#endif

const int threadsPerBlock = 256;
const int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;

__global__ void dot(float *a, float *b, float *c)
{
  __shared__ float cache[threadsPerBlock];

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;

  float temp = 0;

  while(tid < N) {

    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
    
  }

  cache[cacheIndex] = temp;

  __syncthreads();

  // Reduction: threadsPerBlock has to be a power of 2
  int i = blockDim.x/2;
  while(i != 0) {
    
    if (cacheIndex < i)
    {
      cache[cacheIndex] += cache[cacheIndex + i]; 
    }
    __syncthreads();
    i /= 2;

  }

  if (cacheIndex == 0)
    c[blockIdx.x] = cache[0];

}

int main( void ) {

    float   *a, *b, c, *partial_c;
    float   *dev_a, *dev_b, *dev_partial_c;

    a = (float*)malloc( N*sizeof(float) );
    b = (float*)malloc( N*sizeof(float) );
    partial_c = (float*)malloc( blocksPerGrid*sizeof(float) );

    // allocate the memory on the GPU
    checkCudaErrors( cudaMalloc( (void**)&dev_a,N*sizeof(float) ) );
    checkCudaErrors( cudaMalloc( (void**)&dev_b,N*sizeof(float) ) );
    checkCudaErrors( cudaMalloc( (void**)&dev_partial_c,blocksPerGrid*sizeof(float) ) );

    for (int i=0; i<N; i++)
    {
      a[i] = i;
      b[i] = i*2;
    }

    // copy the arrays 'a' and 'b' to the GPU
    checkCudaErrors( cudaMemcpy( dev_a, a, N*sizeof(float),cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy( dev_b, b, N*sizeof(float),cudaMemcpyHostToDevice ) ); 

    dot<<<blocksPerGrid,threadsPerBlock>>>( dev_a, dev_b, dev_partial_c );

    // copy the array 'c' back from the GPU to the CPU
    checkCudaErrors( cudaMemcpy( partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost ) );

    // add the partial sums on the CPU
    c = 0;
    for (int i=0; i<blocksPerGrid; i++)
    {
      c += partial_c[i];
    }

#define sum_squares(x)  (x*(x+1)*(2*x+1)/6)

    printf( "Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares( (float)(N - 1) ) );

    // release GPU memory
    checkCudaErrors( cudaFree( dev_a ) );
    checkCudaErrors( cudaFree( dev_b ) );
    checkCudaErrors( cudaFree( dev_partial_c ) );

    // release CPU memory
    free( a );
    free( b );
    free( partial_c );

}

