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
    float eps = 1e-4; // Example tolerance value




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
    dim3 blocks = dim3((MATRIX_N_width + TILE_WIDTH - 1) / threads.x, (MATRIX_M_height + TILE_WIDTH - 1) / threads.y);

    //NOTE: WIDTH IS X and Y is HEIGHT. So Matrix N's width is P's width and M's height is P's height





    // create cuda event handles
    cudaEvent_t start[4];
    cudaEvent_t stop[4];
    float* gpu_time = (float*)malloc(4 * sizeof(float));/*Allocate 4 floating point numbers for each time instance*/

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
    MatrixMulKernel <<<blocks, threads, 0, 0 >>> (d_M, d_N, d_P, MATRIX_M_width, MATRIX_M_height, MATRIX_N_width);

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

    const int TILE_WIDTH = 16;
    /*height x width*/
    const int matrix_sizes[5][2] = { {100, 100}, {250, 250}, {500, 500}, {1000, 1000}, {1500, 1500} };
    const int num_tests = 3; // Number of tests to perform
    float results[5][4][num_tests]; // [Matrix Size][Timing Type][Test Number]


    /*PART 1 and 2 for discussion questions (change matrix sizes)~~~~~~~~~~~~~*/
    int test;
    for (test = 0; test < num_tests; test++) {
        for (int sizeIndex = 0; sizeIndex < 5; sizeIndex++) {
            float* currentResults = init(matrix_sizes[sizeIndex][0], matrix_sizes[sizeIndex][1], matrix_sizes[sizeIndex][0], matrix_sizes[sizeIndex][1], TILE_WIDTH);
            for (int timingType = 0; timingType < 4; timingType++) {
                results[sizeIndex][timingType][test] = currentResults[timingType];
            }
            free(currentResults); // Assuming init dynamically allocates the returned array
        }
    }

    float averages[5][4], errors[5][4];

    for (int sizeIndex = 0; sizeIndex < 5; sizeIndex++) {
        for (int timingType = 0; timingType < 4; timingType++) {
            float sum = 0, sumSq = 0;
            for (int test = 0; test < num_tests; test++) {
                float value = results[sizeIndex][timingType][test];
                sum += value;
                sumSq += value * value;
            }
            float mean = sum / num_tests;
            averages[sizeIndex][timingType] = mean;
            float variance = (sumSq / num_tests) - (mean * mean);
            errors[sizeIndex][timingType] = sqrt(variance); // Standard deviation as error
        }
    }
    //plotting~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    float home_to_device[5];
    float home_to_device_errors[5];

    float device_to_home[5];
    float device_to_home_errors[5];

    float GPU_runtime[5];
    float GPU_runtime_errors[5];

    float CPU_runtime[5];
    float CPU_runtime_errors[5];


    // Assuming averages[5][4] is filled with the calculated average timings
    for (int i = 0; i < 5; i++) {
        home_to_device[i] = averages[i][0]; // First timing type
        device_to_home[i] = averages[i][2]; // Second timing type
        GPU_runtime[i] = averages[i][1];    // Third timing type
        CPU_runtime[i] = averages[i][3];    // Fourth timing type

        home_to_device_errors[i] = errors[i][0];
        device_to_home_errors[i] = errors[i][2];
        GPU_runtime_errors[i] = errors[i][1];
        CPU_runtime_errors[i] = errors[i][3];
    }



    //PLOT EACH HERE
    printf("Home to Device:\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", home_to_device[i]);
    }
    printf("\n");

    printf("Device to Home:\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", device_to_home[i]);
    }
    printf("\n");

    printf("GPU Runtime:\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", GPU_runtime[i]);
    }
    printf("\n");

    printf("CPU Runtime:\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", CPU_runtime[i]);
    }
    printf("\n");

    /*Part 3 for discussion questions (change tile width)~~~~~~~~~~~~~~~~~~~~~*/

    const int TILE_WIDTHS[5] = {2, 5, 10, 25, 50};
    const int matrix_size[2] = {250, 250};
    float GPU_times[5][3];
    
    for (test = 0; test < num_tests; test++)
    {
        for(int width_index = 0; width_index < 5; width_index++)
        {
            float* currentResults = init(matrix_size[0], matrix_size[1], matrix_size[0], matrix_size[1], TILE_WIDTHS[width_index]);
            GPU_times[width_index][test] = currentResults[1];
            
        }
    }

    float GPU_time_averages[5];
    float GPU_time_errors[5];

    for (int width_index = 0; width_index < 5; width_index++) {
    float sum = 0, sumSq = 0;
    for (int test = 0; test < num_tests; test++) {
        sum += GPU_times[width_index][test];
        sumSq += GPU_times[width_index][test] * GPU_times[width_index][test];
    }
    
        float average = sum / num_tests;
        GPU_time_averages[width_index] = average;
        
        float variance = (sumSq / num_tests) - (average * average);
        GPU_time_errors[width_index] = sqrt(variance); // Standard deviation as the error
    }

    printf("GPU Time Averages and Errors for Different Tile Widths:\n");

    for (int width_index = 0; width_index < 5; width_index++) {
        printf("Tile Width = %d: Average = %f, Error = %f\n", 
            TILE_WIDTHS[width_index], GPU_time_averages[width_index], GPU_time_errors[width_index]);
    }










    /*PLOTTING DATA NOW~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    FILE *file = fopen("timing_data_with_errors.csv", "w");
    if (file == NULL) {
        printf("Error opening file\n");
        return 1;
    }

    // Write headers
    fprintf(file, "TileWidth,HomeToDevice,DeviceToHome,GPU,CPU,HTDError,DTHError,GPUError,CPUError\n");

    // Write data for matrix size change experiments
    for (int i = 0; i < 5; i++) {
        fprintf(file, "%d,%f,%f,%f,%f,%f,%f,%f,%f\n", 
                TILE_WIDTHS[i], 
                home_to_device[i], 
                device_to_home[i], 
                GPU_runtime[i], 
                CPU_runtime[i], 
                home_to_device_errors[i], 
                device_to_home_errors[i], 
                GPU_runtime_errors[i], 
                CPU_runtime_errors[i]);
    }

    // Separate section or additional file for tile width change experiment
    fprintf(file, "\nTileWidth,GPUAverage,GPUError\n");
    for (int i = 0; i < 5; i++) {
        fprintf(file, "%d,%f,%f\n", 
                TILE_WIDTHS[i], 
                GPU_time_averages[i], 
                GPU_time_errors[i]);
    }

    fclose(file);
    printf("Data exported to timing_data_with_errors.csv\n");

    return 0;
}