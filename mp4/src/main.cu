#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define KERNEL_WIDTH       3
#define KERNEL_RADIUS      (KERNEL_WIDTH / 2)
#define TILE_WIDTH         8
#define PADDED_TILE_WIDTH  (TILE_WIDTH + KERNEL_WIDTH - 1)

//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[KERNEL_WIDTH * KERNEL_WIDTH * KERNEL_WIDTH];

__global__ void conv3d(
  float *input, float *output,
  const int z_size, const int y_size, const int x_size
) {
  //@@ Insert kernel code here
  __shared__ float tile[PADDED_TILE_WIDTH][PADDED_TILE_WIDTH][PADDED_TILE_WIDTH];

  int tmp;

  // some alias
  int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
  int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;

  // destination linear index
  int dst = (tz * TILE_WIDTH * TILE_WIDTH) + (ty * TILE_WIDTH) + tx;
  // 3D index inside a padded tiles
  tmp = dst;
  int dst_x = tmp % PADDED_TILE_WIDTH;
  tmp /= PADDED_TILE_WIDTH;
  int dst_y = tmp % PADDED_TILE_WIDTH;
  tmp /= PADDED_TILE_WIDTH;
  int dst_z = tmp;
  // 3D index in global array, simply subtract the pad size
  int src_x = (bx * TILE_WIDTH + dst_x) - KERNEL_RADIUS;
  int src_y = (by * TILE_WIDTH + dst_y) - KERNEL_RADIUS;
  int src_z = (bz * TILE_WIDTH + dst_z) - KERNEL_RADIUS;
  int src = (src_z * y_size * x_size) + (src_y * x_size) + src_x;

  // load 1, this include "left halos" and "content"
  if (
    ((src_x >= 0) && (src_x < x_size)) &&
    ((src_y >= 0) && (src_y < y_size)) &&
    ((src_z >= 0) && (src_z < z_size))
  ) {
    tile[dst_z][dst_y][dst_x] = input[src];
  } else {
    tile[dst_z][dst_y][dst_x] = 0.0f;
  }

  // load 2, "right halos",
  dst = (tz * TILE_WIDTH * TILE_WIDTH) + (ty * TILE_WIDTH) + tx
        + (TILE_WIDTH * TILE_WIDTH * TILE_WIDTH);
  tmp = dst;
  dst_x = tmp % PADDED_TILE_WIDTH;
  tmp /= PADDED_TILE_WIDTH;
  dst_y = tmp % PADDED_TILE_WIDTH;
  tmp /= PADDED_TILE_WIDTH;
  dst_z = tmp;
  src_x = (bx * TILE_WIDTH + dst_x) - KERNEL_RADIUS;
  src_y = (by * TILE_WIDTH + dst_y) - KERNEL_RADIUS;
  src_z = (bz * TILE_WIDTH + dst_z) - KERNEL_RADIUS;
  src = (src_z * y_size * x_size) + (src_y * x_size) + src_x;
  if (dst_z < PADDED_TILE_WIDTH) {
    if (
      ((src_x >= 0) && (src_x < x_size)) &&
      ((src_y >= 0) && (src_y < y_size)) &&
      ((src_z >= 0) && (src_z < z_size))
    ) {
      tile[dst_z][dst_y][dst_x] = input[src];
    } else {
      tile[dst_z][dst_y][dst_x] = 0.0f;
    }
  }

  __syncthreads();

  // the actual convolution
  float sum = 0;
  for (int k = 0; k < KERNEL_WIDTH; k++) {
    for (int j = 0; j < KERNEL_WIDTH; j++) {
      for (int i = 0; i < KERNEL_WIDTH; i++) {
        sum += tile[tz + k][ty + j][tx + i] *
               deviceKernel[
                 (k * KERNEL_WIDTH * KERNEL_WIDTH) + (j * KERNEL_WIDTH) + i
               ];
      }
    }
  }
  // update the destination 3D index
  dst_x = bx * TILE_WIDTH + tx;
  dst_y = by * TILE_WIDTH + ty;
  dst_z = bz * TILE_WIDTH + tz;
  // restore the linear index in global scope
  dst = (dst_z * y_size * x_size) + (dst_y * x_size) + dst_x;
  if ((dst_x < x_size) && (dst_y < y_size) && (dst_z < z_size)) {
    output[dst] = sum;
  }

  __syncthreads();
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == KERNEL_WIDTH * KERNEL_WIDTH * KERNEL_WIDTH);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  size_t inputSize = x_size * y_size * z_size * sizeof(float);
  wbCheck(cudaMalloc(&deviceInput, inputSize));
  wbCheck(cudaMalloc(&deviceOutput, inputSize));
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  size_t kernelSize = kernelLength * sizeof(float);
  wbCheck(cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelSize));
  // Recall that the first three elements of hostInput are dimensions and
  // do not need to be copied to the gpu
  wbCheck(cudaMemcpy(deviceInput, hostInput+3, inputSize, cudaMemcpyHostToDevice));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
  dim3 gridDim(
    ceil((float)x_size/TILE_WIDTH), ceil((float)y_size/TILE_WIDTH), ceil((float)z_size/TILE_WIDTH)
  );
  //@@ Launch the GPU kernel here
  conv3d<<<gridDim, blockDim>>>(
    deviceInput, deviceOutput, z_size, y_size, x_size
  );
  wbCheck(cudaDeviceSynchronize());
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  wbCheck(cudaMemcpy(hostOutput+3, deviceOutput, inputSize, cudaMemcpyDeviceToHost));
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
