#include <stdio.h>
#include <cuda_runtime.h>

void checkCudaErrors(cudaError_t err, const char *userLabel)
{

  if(cudaSuccess != err)
  {
    fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" at user label \"%s\".\n", err, cudaGetErrorString(err), userLabel); 
    exit(EXIT_FAILURE);
  }

}

int main(void)
{

  int *dev_a;

  checkCudaErrors(cudaMalloc((void**)&dev_a, 10000000000000 * sizeof(int)),"allocating dev_a");

  cudaFree(dev_a);

  return 0;
}

