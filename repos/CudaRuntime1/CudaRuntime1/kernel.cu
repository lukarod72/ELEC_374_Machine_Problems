#include "cuda_runtime.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <random>







int correct_output(float* M, float* N, float* P, int Width, const int P_height, const int P_width)
{
    /*TODO: Make alg for matrix mul*/
    int size = P_height * P_width;
    float* product_matrix = (float*)malloc(size * sizeof(float));
    float eps = 1e-5; // Example tolerance value




    for (int row = 0; row < P_height; row++) {
        for (int col = 0; col < P_width; col++) {
            float P_value = 0;
            for (int index = 0; index < Width; index++) {
                P_value += M[row * Width + index] * N[Width * index + col];

            }
            product_matrix[row * Width + col] = P_value;
        }
    }

    for (int i = 0; i < size; i++) {
        if (fabs(product_matrix[i] - P[i]) > eps) { // Use fabs() for absolute difference
            free(product_matrix);
            return 0; // fail
        }
    }

    free(product_matrix);
    return 1;//pass
}

__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width, int P_height, int P_width)
{

    // Calculate the row index of the P element and M
    int Row = blockIdx.y * blockDim.y + threadIdx.y; /*(BLOCK_index in Y)*(Block Dim in Y) + Current thread in y dim*/
    // Calculate the column index of P element and N
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    //check if the current thread is past the actual sizes of the product matrix
    if (Row < P_height && Col < P_width)
    {
        float P_value = 0;

        for (int k = 0; k < Width; k++) {
            P_value += d_M[Row * Width + k] * d_N[k * Width + Col];
        }
        d_P[Row * Width + Col] = P_value;

    }


}

float* init(const int MATRIX_M_height, const int MATRIX_M_width, const int MATRIX_N_height, const int MATRIX_N_width, int TILE_WIDTH) {
    /*PART 2 - Matrix Multiplication*/

    //FIRST ALLOCATE AND DEFINE MATRIXS
    int size_N = MATRIX_N_height * MATRIX_N_width;
    int size_M = MATRIX_M_height * MATRIX_M_width;
    int size_P = MATRIX_M_height * MATRIX_N_width;

    int nbytes_N = size_N * sizeof(float);
    int nbytes_M = size_M * sizeof(float);
    int nbytes_P = size_P * sizeof(float);


    float* N = 0;
    float* M = 0;
    float* d_N = 0;
    float* d_M = 0;

    float* P = 0;
    float* d_P = 0;



    //allocate on HOST
    cudaMallocHost((void**)&N, nbytes_N);
    cudaMallocHost((void**)&M, nbytes_M);
    cudaMallocHost((void**)&P, nbytes_P);

    //Set HOST memory
    int i;
    for (i = 0; i < size_N; i++) { N[i] = (float)rand() / RAND_MAX; }
    for (i = 0; i < size_M; i++) { M[i] = (float)rand() / RAND_MAX; }
    for (i = 0; i < size_P; i++) { P[i] = 0.0f; }





    //allocate and set on DEVICE
    cudaMalloc((void**)&d_N, nbytes_N);
    cudaMalloc((void**)&d_M, nbytes_M);
    cudaMalloc((void**)&d_P, nbytes_P);



    // set kernel launch configuration
    dim3 threads = dim3(TILE_WIDTH, TILE_WIDTH); // Correct: Defines a square block of threads
    dim3 blocks = dim3((MATRIX_N_width + TILE_WIDTH - 1) / threads.x,
        (MATRIX_M_height + TILE_WIDTH - 1) / threads.y);

    //NOTE: WIDTH IS X and Y is HEIGHT. So Matrix N's width is P's width and M's height is P's height





    // create cuda event handles
    cudaEvent_t start[4];
    cudaEvent_t stop[4];
    float* gpu_time = (float*)malloc(4 * sizeof(float));

    for (i = 0; i < 4; i++) {
        cudaEventCreate(&(start[i]));
        cudaEventCreate(&(stop[i]));
    }
    cudaDeviceSynchronize();


    cudaEventRecord(start[0], 0);
    //Copy data from host to device
    cudaMemcpyAsync(d_M, M, nbytes_M, cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_N, N, nbytes_N, cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_P, P, nbytes_P, cudaMemcpyHostToDevice, 0);

    cudaEventRecord(stop[0], 0);
    cudaEventSynchronize(stop[0]); // stop is updated here
    cudaEventElapsedTime(&(gpu_time[0]), start[0], stop[0]); //time difference between start and stop



    // asynchronously issue work to the GPU (all to stream 0)
    cudaEventRecord(start[1], 0);
    //do muliplication
    MatrixMulKernel << <blocks, threads, 0, 0 >> > (d_M, d_N, d_P, MATRIX_M_width, MATRIX_M_height, MATRIX_N_width);

    //stop counter
    cudaEventRecord(stop[1], 0);
    cudaEventSynchronize(stop[1]); // stop is updated here
    cudaEventElapsedTime(&(gpu_time[1]), start[1], stop[1]); //time difference between start and stop

    cudaEventRecord(start[2], 0);
    //Retreive Data from DEVICE
    cudaMemcpyAsync(M, d_M, nbytes_M, cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(N, d_N, nbytes_N, cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(P, d_P, nbytes_P, cudaMemcpyDeviceToHost, 0);
    //              (Destination, Start, Function, Stream)
    cudaEventRecord(stop[2], 0);
    cudaEventSynchronize(stop[2]); // stop is updated here
    cudaEventElapsedTime(&(gpu_time[2]), start[2], stop[2]); //time difference between start and stop







    // have CPU do some work while waiting for GPU to finish
    unsigned long int counter = 0;
    while (cudaEventQuery(stop[2]) == cudaErrorNotReady)
    {
        counter++; // Indicates that the CPU is running asynchronously while GPU is executing
    }



    cudaEventRecord(start[3], 0);

    // check the output for correctness
    bool bFinalResults = (bool)correct_output(M, N, P, MATRIX_M_width, MATRIX_M_height, MATRIX_N_width);
    if (bFinalResults) {
        printf("\nTest PASSED\n");
    }
    else {
        printf("\nTest FAILED\n");

    }

    cudaEventRecord(stop[3], 0);
    cudaEventSynchronize(stop[3]); // stop is updated here
    cudaEventElapsedTime(&(gpu_time[3]), start[3], stop[3]); //time difference between start and stop



    //managing time instances
    for (i = 0; i < 4; i++)
    {
        // release resources
        cudaEventDestroy(start[i]);
        cudaEventDestroy(stop[i]);
    }





    //FREE M
    cudaFreeHost(M);
    cudaFree(d_M);

    //FREE N
    cudaFreeHost(N);
    cudaFree(d_N);

    //FREE P
    cudaFreeHost(P);
    cudaFree(d_P);


    cudaDeviceReset();

    return gpu_time;
}
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

    const int TILE_WIDTH = 10;
    /*height x width*/
    const int matrix_sizes[5][2] = { {100, 100}, {250, 250}, {500, 500}, {1000, 1000}, {1500, 1500} };
    float** values = (float**)malloc(5 * sizeof(float*));
    int i;
    for (i = 0; i < 5; i++)
    {

        printf("Matrix: %d\n", i);
        values[i] = init(matrix_sizes[i][0], matrix_sizes[i][1], matrix_sizes[i][0], matrix_sizes[i][1], TILE_WIDTH);
        printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    }

    printf("\n\nTIME RESULTS: ");
    for (i = 0; i < 5; i++) {
        for (int j = 0; j < 4; j++)
        {
            printf("%f, ", values[i][j]);
        }
        printf("\n");
    }



    //plotting~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    float home_to_device[5];
    float device_to_home[5];
    float GPU_runtime[5];
    float CPU_runtime[5];

    for (i = 0; i < 5; i++)
    {
        home_to_device[i] = values[i][0];
        device_to_home[i] = values[i][2];
        GPU_runtime[i] = values[i][1];
        CPU_runtime[i] = values[i][3];
        free(values[i]);
    }
    //PLOT EACH HERE



    free(values);
    return 0;
}
