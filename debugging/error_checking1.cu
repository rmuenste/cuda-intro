#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{

  int *dev_a;

  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  err = cudaMalloc((void**)&dev_a, 10000000000000 * sizeof(int));

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector, error message: \"%s\"!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  cudaFree(dev_a);

  return 0;
}


