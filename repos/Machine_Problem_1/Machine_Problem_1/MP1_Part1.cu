#include "cuda_runtime.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>



int main(int argc, char* argv[])
{

    /*PART 1 - Device Query Information*/
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("Device %d: %s\n", dev, deviceProp.name);
        printf("  CUDA Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Clock Rate: %d kHz\n", deviceProp.clockRate);
        printf("  Number of SMs: %d\n", deviceProp.multiProcessorCount);
        // Number of cores is a function of CUDA capability, which varies by device. This requires a lookup based on the specific architecture.
        printf("  Warp Size: %d\n", deviceProp.warpSize);
        printf("  Global Memory: %lu bytes\n", deviceProp.totalGlobalMem);
        printf("  Constant Memory: %lu bytes\n", deviceProp.totalConstMem);
        printf("  Shared Memory Per Block: %lu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Registers Per Block: %d\n", deviceProp.regsPerBlock);
        printf("  Max Threads Per Block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max Block Dimensions: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Max Grid Dimensions: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("\n");
    }

    if (deviceCount == 0) {
        printf("No CUDA-compatible device found\n");
    }

   
    return 0;
}