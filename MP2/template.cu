
#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  float value = 0;
  if(idx < numCRows && idy < numCColumns){
    for(int j = 0; j < numAColumns; j ++){
      value += A[idx*numAColumns + j] * B[j*numBColumns + idy];
    }
    C[idx*numCColumns + idy] = value;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(sizeof(float)*numCRows*numCColumns);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  wbCheck(cudaMalloc((void **)&deviceA, sizeof(float)*numARows*numAColumns));
  wbCheck(cudaMalloc((void **)&deviceB, sizeof(float)*numBRows*numBColumns));
  wbCheck(cudaMalloc((void **)&deviceC, sizeof(float)*numCRows*numCColumns));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  wbCheck(cudaMemcpy(deviceA, hostA, sizeof(float)*numARows*numAColumns, cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceB, hostB, sizeof(float)*numBRows*numBColumns, cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  auto ceil = [](int a, int b){
    return int((a + b - 1) / b);
  };
  int BLOCK_SIZE_X = 8;
  int BLOCK_SIZE_Y = 8;
  //int BLOCK_SIZE = BLOCK_SIZE_X * BLOCK_SIZE_Y;
  dim3 DimGrid(ceil(numCRows*numCColumns, BLOCK_SIZE_X), 
    ceil(numCRows*numCColumns, BLOCK_SIZE_Y), 1);
  dim3 DimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, 
    numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  wbCheck(cudaMemcpy(hostC, deviceC, sizeof(float)*numCRows*numCColumns, cudaMemcpyDeviceToHost));
  /*
  float verifyC[numCRows*numCColumns];
  for(int i=0;i<numCRows;++i){
    for(int j=0;j<numCColumns; ++j){
      float res = 0;
      for(int k=0;k<numAColumns;++k){
          res += hostA[i*numAColumns +k] * hostB[k*numBColumns+ j];
      }
      verifyC[i*numCColumns+j] = res;
    }
  }
  for(int i=0;i<numCRows;++i){
    for(int j=0;j<numCColumns; ++j){
      if(abs(verifyC[i*numCColumns+j] - hostC[i*numCColumns+j]) >= 1e-6){
        wbLog(TRACE, "expect ", verifyC[i*numCColumns+j]," but get ", hostC[i*numCColumns+j] , " in ", i, " ", j);
      }
    }
  }
  */

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
