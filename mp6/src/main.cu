// Histogram Equalization

#include <cassert>
#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define HISTOGRAM_LENGTH 256
__constant__ float constCdf[HISTOGRAM_LENGTH];

typedef unsigned char uchar;

//@@ insert code here
__global__ void float2uchar(uchar *out, float *in, size_t n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = (uchar)(in[i] * 255.0);
  }
}

__global__ void rgb2gray(uchar *gray, uchar *rgb, size_t n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    uchar r = rgb[3*i];
    uchar g = rgb[3*i + 1];
    uchar b = rgb[3*i + 2];
    gray[i] = (uchar)(0.21*r + 0.71*g + 0.07*b);
  }
}

/** histogram, v1, (id=5, 0.163981ms)
__global__ void histogram(unsigned int *histogram, uchar *image, size_t n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(&histogram[image[i]], 1);
  }
}
**/
/** histogram, v2, (id=5, 2.483228ms) **/
__global__ void histogram(unsigned int *histogram, uchar *image, size_t n) {
  __shared__ unsigned int private_hist[HISTOGRAM_LENGTH];

  unsigned int t = threadIdx.x;

  // clear shared memory
  if (t < HISTOGRAM_LENGTH) {
    private_hist[t] = 0;
  }
  __syncthreads();

  unsigned int stride = blockDim.x * gridDim.x;
  for (unsigned int i = t; i < n; i += stride) {
    atomicAdd(&private_hist[image[i]], 1);
  }
  __syncthreads();

  // add back to global
  if (t < HISTOGRAM_LENGTH) {
    histogram[t] += private_hist[t];
  }
}

__device__ float cdfMin;

__device__ void findCdfMin(float *cdf) {
  cdfMin = 0;
  for (unsigned int i = 0; i < HISTOGRAM_LENGTH-1; i++) {
    if ((cdf[i] == 0) && (cdf[i+1] > 0)) {
      cdfMin = cdf[i+1];
      break;
    }
  }
}

__global__ void hist2cdf(float *cdf, unsigned int *hist, size_t n) {
  __shared__ unsigned int smem[HISTOGRAM_LENGTH];

  // load histogram to shared memory
  for (unsigned int i = threadIdx.x; i < HISTOGRAM_LENGTH; i += blockDim.x) {
    smem[i] = hist[i];
  }

  // start scanning operation
  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
    __syncthreads();

    unsigned i = 2 * stride * (threadIdx.x + 1) - 1;
    if (i < HISTOGRAM_LENGTH) {
      smem[i] += smem[i - stride];
    }
  }

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    __syncthreads();

    unsigned int i = 2 * stride * (threadIdx.x + 1) - 1;
    if (i + stride < HISTOGRAM_LENGTH) {
      smem[i + stride] += smem[i];
    }
  }

  // store the cdf back
  for (unsigned int i = threadIdx.x; i < HISTOGRAM_LENGTH; i += blockDim.x) {
    cdf[i] = (float)smem[i] / n;
  }

  /** without this, (id=5, 0.026754ms) **/
  /** (match correctColor, v3), (id=5, 0.077918) **/
  if (threadIdx.x == 0) {
    findCdfMin(constCdf);
  }
}

__device__ float clamp(float value, float min_value, float max_value) {
  return min(max(value, min_value), max_value);
}

/** correctColor, v1, (id=5, 0.775989ms)
__global__ void correctColor(uchar *out, uchar *in, float *cdf, size_t n) {
  if (threadIdx.x == 0) {
    findCdfMin(cdf);
  }
  __syncthreads();

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float value = 255 * (cdf[in[i]] - cdfMin) / (1 - cdfMin);
    out[i] = clamp(value, 0, 255);
  }
}
**/
/** correctColor, v2, (id=5, 0.559565ms)
__global__ void correctColor(uchar *out, uchar *in, size_t n) {
  if (threadIdx.x == 0) {
    findCdfMin(constCdf);
  }
  __syncthreads();

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float value = 255 * (constCdf[in[i]] - cdfMin) / (1 - cdfMin);
    out[i] = clamp(value, 0, 255);
  }
}
**/
/** correctColor, v3, (id=5, 0.396279ms) **/
__global__ void correctColor(uchar *out, uchar *in, size_t n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float value = 255 * (constCdf[in[i]] - cdfMin) / (1 - cdfMin);
    out[i] = clamp(value, 0, 255);
  }
}

__global__ void uchar2float(float *out, uchar *in, size_t n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = (float)in[i] / 255;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImageData;
  float *deviceOutputImageData;
  uchar *deviceRgbImage;
  uchar *deviceGrayImage;
  unsigned int *deviceHistogram;
  float *deviceCdf;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  wbTime_start(GPU, "Allocating GPU memory.");
  size_t numPixels = imageWidth * imageHeight;
  wbCheck(cudaMalloc(&deviceInputImageData,
                     numPixels * imageChannels * sizeof(float)));
  wbCheck(cudaMalloc(&deviceOutputImageData,
                     numPixels * imageChannels * sizeof(float)));
  wbCheck(cudaMalloc(&deviceRgbImage,
                     numPixels * imageChannels * sizeof(uchar)));
  wbCheck(cudaMalloc(&deviceGrayImage,
                     numPixels * sizeof(uchar)));
  wbCheck(cudaMalloc(&deviceHistogram,
                     HISTOGRAM_LENGTH * sizeof(unsigned int)));
  wbCheck(cudaMalloc(&deviceCdf,
                     HISTOGRAM_LENGTH * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData,
                     numPixels * imageChannels * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  dim3 linearBlock(1024);
  dim3 linearRgbGrid(ceil((float)numPixels * imageChannels / 1024));
  dim3 linearGrayGrid(ceil((float)numPixels / 1024));

  // sanity check for histogram assumption
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  assert((HISTOGRAM_LENGTH <= deviceProp.maxThreadsPerBlock)
         && "histogram length exceeds maximum threads per block");

  wbTime_start(Compute, "Performing CUDA computation");

  float2uchar<<<linearRgbGrid, linearBlock>>>(deviceRgbImage,
                                              deviceInputImageData,
                                              numPixels * imageChannels);

  wbTime_start(Compute, "Convert the image from RGB to grayscale");
  rgb2gray<<<linearGrayGrid, linearBlock>>>(deviceGrayImage,
                                            deviceRgbImage,
                                            numPixels);
  wbTime_stop(Compute, "Convert the image from RGB to grayscale");
  wbTime_start(Compute, "Compute the histogram");
  wbCheck(cudaMemset(deviceHistogram, 0,
                     HISTOGRAM_LENGTH * sizeof(unsigned int)));
  /** histogram, v1, (id=5, 0.163981ms)
  histogram<<<linearGrayGrid, linearBlock>>>(deviceHistogram,
                                             deviceGrayImage,
                                             numPixels);
  **/
  histogram<<<1, linearBlock>>>(deviceHistogram,
                                deviceGrayImage,
                                numPixels);
  wbTime_stop(Compute, "Compute the histogram");
  wbTime_start(Compute, "Compute the cumulative distribution dunction");
  // assume histogram length <= maximum threads per block
  hist2cdf<<<1, ceil((float)HISTOGRAM_LENGTH / 2)>>>(deviceCdf,
                                                     deviceHistogram,
                                                     numPixels);
  wbTime_stop(Compute, "Compute the cumulative distribution dunction");
  wbTime_start(Compute, "Color correction");
  /** correctColor, v1, (id=5, 0.775989ms)
  correctColor<<<linearRgbGrid, linearBlock>>>(deviceRgbImage,
                                               deviceRgbImage,
                                               deviceCdf,
                                               numPixels * imageChannels);
  **/
  cudaMemcpyToSymbol(constCdf, deviceCdf,
                     HISTOGRAM_LENGTH * sizeof(float),
                     0,
                     cudaMemcpyDeviceToDevice);
  correctColor<<<linearRgbGrid, linearBlock>>>(deviceRgbImage,
                                               deviceRgbImage,
                                               numPixels * imageChannels);
  wbTime_stop(Compute, "Color correction");

  uchar2float<<<linearRgbGrid, linearBlock>>>(deviceOutputImageData,
                                              deviceRgbImage,
                                              numPixels * imageChannels);
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData,
                     numPixels * imageChannels * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbSolution(args, outputImage);

  //@@ insert code here
  wbTime_start(GPU, "Freeing GPU Memory");
  wbCheck(cudaFree(deviceInputImageData));
  wbCheck(cudaFree(deviceOutputImageData));
  wbCheck(cudaFree(deviceRgbImage));
  wbCheck(cudaFree(deviceGrayImage));
  wbCheck(cudaFree(deviceHistogram));
  wbCheck(cudaFree(deviceCdf));
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbImage_delete(inputImage);
  wbImage_delete(outputImage);

  return 0;
}
