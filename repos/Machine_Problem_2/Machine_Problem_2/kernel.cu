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
    float eps = 1e-3; // Example tolerance value




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

__global__ void MatrixMul_Tiled(float* d_M, float* d_N, float* d_P, int Width, int P_height, int P_width, int TILE_WIDTH) {
    extern __shared__ float shared[];
    float* tile_M = shared;
    float* tile_N = tile_M + TILE_WIDTH * TILE_WIDTH;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = blockIdx.y * blockDim.y + ty;
    int Col = blockIdx.x * blockDim.x + tx;

    if (Row >= P_height || Col >= P_width) return;  // Early exit for out-of-bounds threads

    float P_value = 0.0;

    for (int m = 0; m < (Width - 1) / TILE_WIDTH + 1; ++m) {
        int M_index = Row * Width + m*TILE_WIDTH + tx;
        int N_index = (m*TILE_WIDTH + ty) * Width + Col;

        // Load tile_M and tile_N with boundary checks
        if (m*TILE_WIDTH + tx < Width)
            tile_M[ty * TILE_WIDTH + tx] = d_M[M_index];
        else
            tile_M[ty * TILE_WIDTH + tx] = 0.0;

        if (m*TILE_WIDTH + ty < Width)
            tile_N[ty * TILE_WIDTH + tx] = d_N[N_index];
        else
            tile_N[ty * TILE_WIDTH + tx] = 0.0;

        __syncthreads();

        // Perform tile multiplication
        for (int k = 0; k < TILE_WIDTH; ++k) {
            if (m*TILE_WIDTH + k < Width) // Boundary check for partial tiles
                P_value += tile_M[ty * TILE_WIDTH + k] * tile_N[k * TILE_WIDTH + tx];
        }
        __syncthreads();
    }

    d_P[Row * P_width + Col] = P_value;  // Write the computed value back to global memory
}



float* init(const int MATRIX_M_height, const int MATRIX_M_width, const int MATRIX_N_height, const int MATRIX_N_width, const int TILE_WIDTH) {
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

    size_t sharedMemSize = 2*TILE_WIDTH*TILE_WIDTH*sizeof(float);

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
    MatrixMul_Tiled <<<blocks, threads, sharedMemSize, 0 >>> (d_M, d_N, d_P, MATRIX_M_width, MATRIX_M_height, MATRIX_N_width, TILE_WIDTH);

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
        printf("\nTest PASSED: TILE_WIDTH:%d, Matrix_size:%d\n", TILE_WIDTH, MATRIX_N_width);
    }
    else {
        printf("\nTest FAILED: TILE_WIDTH:%d, Matrix_size:%d\n", TILE_WIDTH, MATRIX_N_width);

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


    /*PART 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    const int matrix_sizes[5][2] = { {100, 100}, {250, 250}, {500, 500}, {1000, 1000}, {1500, 1500} };
    const int num_tests = 3; // Number of tests to perform (this is for plotting later)
    const int TILE_WIDTHS[5] = {2, 5, 10, 25, 50};
    int test, width_index, size_index;
    float GPU_times[5][5][3];//[width_index][size_index][test]
    
    for(width_index = 0; width_index < 5; width_index++){

        for(size_index = 0; size_index < 5; size_index++){

            for(test = 0; test < num_tests; test++){
                
                //capture output of init (ie the kernel setup) into local variable
                float* currentResults = init(matrix_sizes[size_index][0], matrix_sizes[size_index][1], matrix_sizes[size_index][0], matrix_sizes[size_index][1], TILE_WIDTHS[width_index]);
                GPU_times[width_index][size_index][test] = currentResults[1];//only take the second element (the time required to complete using GPU)

            }

        }

    }

    float GPU_time_averages[5][5];//[width_index][size_index]
    float GPU_time_errors[5][5];//[width_index][size_index]

    for (width_index = 0; width_index < 5; width_index++) {
        for(size_index = 0; size_index < 5; size_index++){
            float sum = 0, sumSq = 0;
            for (int test = 0; test < num_tests; test++) {
                sum += GPU_times[width_index][size_index][test];
                sumSq += GPU_times[width_index][size_index][test] * GPU_times[width_index][size_index][test];
            }
        
            float average = sum / num_tests;
            GPU_time_averages[width_index][size_index] = average;
            
            float variance = (sumSq / num_tests) - (average * average);
            GPU_time_errors[width_index][size_index] = sqrt(variance); // Standard deviation as the error
        }
    }

    printf("GPU Time Averages and Errors for Different Tile Widths:\n");

    for (width_index = 0; width_index < 5; width_index++) {
        printf("\nTile Width = %d:~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n", TILE_WIDTHS[width_index]);

        for(size_index = 0; size_index < 5; size_index++){
            printf("Matrix Size = {%dx%d}: Average = %f, Error = %f\n", matrix_sizes[size_index][0], matrix_sizes[size_index][1], GPU_time_averages[width_index][size_index], GPU_time_errors[width_index][size_index]);

        }
    }










    /*PLOTTING DATA NOW for CSV file~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    const int sizes[5] = {100, 250, 500, 1000, 1500}; // Matrix widths for example


    //Question 1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    FILE *file = fopen("timing_data_with_errors.csv", "w");
    if (file == NULL) {
        printf("Error opening file\n");
        return 1;
    }


    file = fopen("tile_width_timing_data.csv", "w");
    if (file == NULL) {
        printf("Error opening file\n");
        return 1;
    }
    fprintf(file, "TileWidth,MatrixSize,GPUAverage,GPUError\n");
    for(width_index = 0; width_index < 5; width_index++){
        
        for(size_index = 0; size_index < 5; size_index++){
            fprintf(file, "%d,%d,%f,%f\n",
                TILE_WIDTHS[width_index],
                matrix_sizes[size_index][0],
                GPU_time_averages[width_index][size_index],
                GPU_time_errors[width_index][size_index]);
        }

    }

    fclose(file);
    printf("Data exported to tile_width_timing_data.csv\n");

    return 0;
}