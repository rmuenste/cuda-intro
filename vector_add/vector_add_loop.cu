#include <stdio.h>
#include <cuda_runtime.h>

#define N (32 * 1024)

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
__global__ void add(int *a, int *b, int *c)
{

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  while (tid < N)
  {
    c[tid] = a[tid] + b[tid];
    // blockDim.x: number of threads in x-blocks
    // gridDim.x : number of blocks in x-grid
    tid += blockDim.x * gridDim.x;
  }

}

int main(void)
{

  int *a, *b, *c; 
  int *dev_a, *dev_b, *dev_c;

  // allocate the memory on the CPU
  a = (int*)malloc( N * sizeof(int) );
  b = (int*)malloc( N * sizeof(int) );
  c = (int*)malloc( N * sizeof(int) );

  checkCudaErrors(cudaMalloc((void**)&dev_a, N * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&dev_b, N * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&dev_c, N * sizeof(int)));

  for (int i = 0; i < N; ++i)
  {
    a[i] = i; 
    b[i] = i * 2; 
  }

  checkCudaErrors(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

  int threadsPerBlock = 128;
  int blocksPerGrid = 128; 
  add<<<blocksPerGrid,threadsPerBlock>>>(dev_a, dev_b, dev_c);

  checkCudaErrors(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost)); 

  for (int i = 0; i < N; ++i)
  {
    printf("%d + %d = %d\n",a[i],b[i],c[i]); 
  }

  // verify that the GPU did the work we requested

  bool success = true;
  for (int i=0; i<N; i++)
  {
    if ((a[i] + b[i]) != c[i])
    {
      printf( "Error:  %d + %d != %d\n", a[i], b[i], c[i] );
      success = false;
    }
  }
  if (success)    printf( "We did it!\n" );

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  free(a);
  free(b);
  free(c);

  return 0;
}
