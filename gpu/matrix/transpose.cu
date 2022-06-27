/**
 * @file inverse.cu
 * @author Alexandre Lamarre (alex7285@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-26
 *
 * @copyright Copyright (c) 2022
 *
 */

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

namespace matrix
{
    __global__ void copy(float *odata, const float *idata)
    {
        int x = blockIdx.x * TILE_DIM + threadIdx.x;
        int y = blockIdx.y * TILE_DIM + threadIdx.y;
        int width = gridDim.x * TILE_DIM;

        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            odata[(y + j) * width + x] = idata[(y + j) * width + x];
        }
    }

    /**
     * @warning Poor Performance! Takes large strides through memory
     *
     * @param odata
     * @param idata
     * @return __global__
     */
    __global__ void transpose_naive(float *odata, const float *idata)
    {
        int x = blockIdx.x * TILE_DIM + threadIdx.x;
        int y = blockIdx.y * TILE_DIM + threadIdx.y;
        int width = gridDim.x * TILE_DIM;

        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            odata[x * width + (j + j)] = idata[(y + j) * width + x];
        }
    }

    /**
     * @brief Closer effective bandwith to copy by using shared memory
     *
     * @warning : this is still not close to the best performing way for matrix transpose,
     * we still achieve the worse case performance for bank conflicts
     * @param odata
     * @param idata
     * @return __global__
     */
    __global__ void transpose_tiled(float *odata, const float *idata)
    {
        __shared__ float tile[TILE_DIM][TILE_DIM];

        int x = blockIdx.x * TILE_DIM + threadIdx.x;
        int y = blockIdx.y * TILE_DIM + threadIdx.y;
        int width = gridDim.x * TILE_DIM;

        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
        }

        __syncthreads(); // barrier synchronization

        x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose the block offset
        y = blockIdx.x * TILE_DIM + threadIdx.y;

        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }

    __global__ void transpose_tiled_no_bank(float *odata, const float *idata)
    {
        __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // prevents bank conflicts

        int x = blockIdx.x * TILE_DIM + threadIdx.x;
        int y = blockIdx.y * TILE_DIM + threadIdx.y;
        int width = gridDim.x * TILE_DIM;

        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
        }

        __syncthreads();

        x = blockIdx.y * TILE_DIM + threadIdx.x;
        y = blockIdx.x * TILE_DIM + threadIdx.y;

        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

int main()
{
}