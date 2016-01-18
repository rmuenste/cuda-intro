#include <stdio.h>
#include <cuda_runtime.h>

#define N 10

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
  int tid = blockIdx.x;
  if (tid < N)
    c[tid] = a[tid] + b[tid];
}

int main(void)
{

  int a[N], b[N], c[N]; 
  int *dev_a, *dev_b, *dev_c;


  checkCudaErrors(cudaMalloc((void**)&dev_a, 10000000000000 * sizeof(int)));

  cudaMalloc((void**)&dev_b, N * sizeof(int));
  cudaMalloc((void**)&dev_c, N * sizeof(int));

  for (int i = 0; i < N; ++i)
  {
    a[i] = -i; 
    b[i] = i * i; 
  }

  cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

  add<<<N,1>>>(dev_a, dev_b, dev_c);

  cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost); 

  for (int i = 0; i < N; ++i)
  {
    printf("%d + %d = %d\n",a[i],b[i],c[i]); 
  }

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 0;
}

