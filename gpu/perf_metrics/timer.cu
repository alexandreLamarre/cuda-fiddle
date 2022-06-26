#include <iostream>
#include <math.h>

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
    int N = 1 << 20;
    float *x, *y;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate unified memory (available on all devices)
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // intialize x and y arrays on the host
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int blockSize = 256;
    int numBLocks = (N + blockSize - 1) / blockSize;

    cudaEventRecord(start);
    add_gpu<<<numBLocks, blockSize>>>(N, x, y);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Elapsed time : " << elapsedTime << " ms" << std::endl;

    std::cout << "Effective bandwith (GB/s) : " << N * 4 * 3 / (elapsedTime / 1e6) << std::endl;
}