#include <stdio.h>
#include <cuda_runtime.h>

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

int main(void)
{

  int *dev_a;

  checkCudaErrors(cudaMalloc((void**)&dev_a, 10000000000000 * sizeof(int)));

  cudaFree(dev_a);

  return 0;
}

