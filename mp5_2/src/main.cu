// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan_block(float *output, float *input, float *blockSum, int n) {
  __shared__ float partialSum[2*BLOCK_SIZE];

  unsigned int t = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;

  if (start+t < n) {
    partialSum[t] = input[start+t];
  } else {
    partialSum[t] = 0;
  }
  if (start+t + BLOCK_SIZE < n) {
    partialSum[t + BLOCK_SIZE] = input[start+t + BLOCK_SIZE];
  } else {
    partialSum[t + BLOCK_SIZE] = 0;
  }

  for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
    __syncthreads();

    unsigned i = 2 * stride * (threadIdx.x + 1) - 1;
    if (i < 2*BLOCK_SIZE) {
      partialSum[i] += partialSum[i - stride];
    }
  }

  for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    __syncthreads();

    unsigned int i = 2 * stride * (threadIdx.x + 1) - 1;
    if (i + stride < 2*BLOCK_SIZE) {
      partialSum[i + stride] += partialSum[i];
    }
  }

  __syncthreads();

  // store (partial) sum
  if (start+t < n) {
    output[start+t] = partialSum[t];
  }
  if (start+t + BLOCK_SIZE < n) {
    output[start+t + BLOCK_SIZE] = partialSum[t + BLOCK_SIZE];
  }

  // store block sum
  if (t == BLOCK_SIZE-1) {
    blockSum[blockIdx.x] = partialSum[2*BLOCK_SIZE-1];
  }
}

__global__ void add_block_sum(float *output, float *blockSum, int n) {
  __shared__ float increment;

  unsigned int t = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;

  // load increment for current block
  if (t == 0) {
    if (blockIdx.x == 0) {
      increment = 0;
    } else {
      increment = blockSum[blockIdx.x-1];
    }
  }

  __syncthreads();

  if (start+t < n) {
    output[start+t] += increment;
  }
  if (start+t + BLOCK_SIZE < n) {
    output[start+t + BLOCK_SIZE] += increment;
  }
}

void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  float *blockSum;

  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim(ceil((float)len / (2*BLOCK_SIZE)));
  wbLog(TRACE, "do scan(), grid.x=", gridDim.x);
  cudaMalloc((void **)&blockSum, gridDim.x * sizeof(float));

  // scan each block
  scan_block<<<gridDim, blockDim>>>(output, input, blockSum, len);
  cudaDeviceSynchronize();
  // scan over block sums
  if (gridDim.x > 1) {
    scan(blockSum, blockSum, gridDim.x);
  }
  // add back accumulated block sums
  add_block_sum<<<gridDim, blockDim>>>(output, blockSum, len);
  cudaDeviceSynchronize();

  cudaFree(blockSum);
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  // grid/block dimension inits are moved to `scan`
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan(deviceInput, deviceOutput, numElements);
  // deal with device synchronization in `scan`
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  wbCheck(cudaFree(deviceInput));
  wbCheck(cudaFree(deviceOutput));
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
