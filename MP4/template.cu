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
#define TILE_WIDTH 2
#define MASK_WIDTH 3
#define BLOCK_WIDTH TILE_WIDTH + MASK_WIDTH - 1
//@@ Define constant memory for device kernel here
__device__ __constant__ float  kernel[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];
__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int offset = MASK_WIDTH / 2;
  // begin in tiled for (idz, idy, idx)
  int tx = threadIdx.x - offset;
  int ty = threadIdx.y - offset;
  int tz = threadIdx.z - offset;
  // index in output and input
  int idx = blockIdx.x * TILE_WIDTH + threadIdx.x - offset;
  int idy = blockIdx.y * TILE_WIDTH + threadIdx.y - offset;
  int idz = blockIdx.z * TILE_WIDTH + threadIdx.z - offset;

  __shared__ float tiled[BLOCK_WIDTH][BLOCK_WIDTH][BLOCK_WIDTH];

  if(idx>=0 && idy>=0 && idz>=0 && idx < x_size && idy < y_size && idz < z_size){
    tiled[threadIdx.z][threadIdx.y][threadIdx.x] = input[idz*x_size*y_size+ idy * x_size + idx];
  }else{
    tiled[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
  }
  __syncthreads();
  float PValues = 0;
  if(idx>=0 && idy>=0 && idz>=0 && idx < x_size && idy < y_size && idz < z_size && 
    tx>=0 && ty>=0 && tz>=0 && tx<TILE_WIDTH && ty< TILE_WIDTH && tz< TILE_WIDTH){
    for(int i=0, z=tz; i < MASK_WIDTH; ++i, ++z){
      for(int j=0, y=ty; j< MASK_WIDTH; ++j, ++y){
        for(int k=0, x=tx; k< MASK_WIDTH; ++k, ++x){
          PValues += tiled[z][y][x] * kernel[i][j][k];
        }
      }
    }
  }
  __syncthreads();
  if(idx>=0 && idy>=0 && idz>=0 && idx < x_size && idy < y_size && idz < z_size && 
    tx>=0 && ty>=0 && tz>=0 && tx<TILE_WIDTH && ty< TILE_WIDTH && tz< TILE_WIDTH){
      output[idz*x_size*y_size+ idy * x_size + idx] = PValues;
  }
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
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  wbCheck(cudaMalloc((void **)&deviceInput, (inputLength - 3) * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, (inputLength - 3) * sizeof(float)));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  wbCheck(cudaMemcpy(deviceInput, hostInput+3, (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpyToSymbol(kernel, hostKernel, kernelLength*sizeof(float)));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  auto ceil = [](int a, int b){
    return (a+b-1)/b;
  };
  
  dim3 DimGrid(ceil(x_size,TILE_WIDTH), ceil(y_size, TILE_WIDTH), ceil(z_size, TILE_WIDTH));
  dim3 DimBlock(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);

  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  wbCheck(cudaDeviceSynchronize());
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbCheck(cudaMemcpy(hostOutput+3, deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost));
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
