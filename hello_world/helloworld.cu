#include <stdio.h>


// Cuda supports printf in kernels for
// hardware with compute compatibility >= 2.0
__global__ void helloworld()
{ 
  // CUDA runtime uses device overloading or printf in kernels
  printf("Hello world!\n");
}

int main(void)
{
  helloworld<<<1,1>>>();  
  return 0;
}
