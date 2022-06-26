/**
 * @file add.cu
 * @author Alexandre Lamarre (alex7285@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-26
 *
 * @copyright Copyright (c) 2022
 *
 * Sample program to add two arrays with a million
 * elements each using a CUDA kernel.
 */

#include <iostream>
#include <math.h>

void add_cpu(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
    {
        x[i] += y[i];
    }
}

__global__ void add_gpu(int n, float *X, float *Y)
{
    int index = threadIdx.x; // contains id of the current thread within the block
    int stride = blockDim.x; // contains the number of threads in the block
    for (int i = index; i < n; i += stride)
    {
        // Note : Y[i] += X[i] is not allowed in CUDA
        Y[i] = X[i] + Y[i];
    }
}

int main(void)
{
    std::cout << "Running add on the CPU..." << std::endl;
    int N = 1 << 20; // 1 million elements

    float *x = new float[N];
    float *y = new float[N];

    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add_cpu(N, x, y);

    // Check the result
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        float error = fabs(x[i] - 3.0f);
        if (error > maxError)
        {
            maxError = error;
        }
    }
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    delete[] x;
    delete[] y;

    std::cout << "Running add on the GPU..." << std::endl;
    int M = 1 << 20; // 1 million elements
    float *X, *Y;

    // Allocate unified memory (available on all devices)
    cudaMallocManaged(&X, M * sizeof(float));
    cudaMallocManaged(&Y, M * sizeof(float));

    // intialize x and y arrays on the host
    for (int i = 0; i < M; i++)
    {
        X[i] = 1.0f;
        Y[i] = 2.0f;
    }
    int blockSize = 256;
    int numBLocks = (M + blockSize - 1) / blockSize;

    // First parameter in cuda kernel is the number of thread blocks
    // Second paramter in cuda kernel is the number of parallel threads to execute on
    add_gpu<<<numBLocks, blockSize>>>(M, X, Y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check the result
    float maxErrorGPU = 0.0f;
    for (int i = 0; i < M; i++)
    {
        maxErrorGPU = fmax(maxErrorGPU, fabs(Y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxErrorGPU << std::endl;

    cudaFree(X);
    cudaFree(Y);
    return 0;
}