#include "cuda_runtime.h"
#include <cuda.h>
#include <time.h>
#include <ctime>
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <algorithm>
using namespace std;
clock_t startSeqFull, endSeqFull, startSeqComp, endSeqComp, startGPUFull, endGPUFull, startGPUcomp, endGPUcomp;
clock_t startKer1, endKer1, startKer2, endKer2, startKer3, endKer3, startKer4, endKer4;
clock_t startKer1Seq, endKer1Seq, startKer2Seq, endKer2Seq, startKer3Seq, endKer3Seq, startKer4Seq, endKer4Seq;
#define ull unsigned long long
#define SHARED_MEM_ARR_SIZE 1024
#define BLOCK_WIDTH 1024
#define SECTION_SIZE 32
#define DEBUG 1
#define TILE_DIM 16
//#define TEST
#define BLOCK_ROWS  16
#define RAND_BOUND 1
//#define DEBUG_SEQ
//#define parallelDEBUG
cudaError_t TheWrapperofWrappers(int* NonZerosCount,  int* M, unsigned int size, int *JDS_row_index, int*X, int *Y, int actualSize, int*arrOfindices);
cudaError_t toELL(int* M, int* Values, int* Columns, int maxRowSize, int size, int maxZeroCount);
cudaError_t spMV_Sectioned_ELL(int *data, int *col_indices, int* NonZerosCount, int* JDS_row_index, int* x, int* y ,int  maxRowSize, int size );
cudaError_t transpose(int* A, int size, int maxRowSize);
//1-> count non-zero(sum reduction).	2-> padKernel.	3->transpose.	 4->SpMV
__global__ void transposeNoBankConflicts(int* odata, int* idata, int width, int height)
{
	__shared__ int tile[TILE_DIM][TILE_DIM + 1];

	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex)*width;

	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex)*height;


	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
	{
		tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
	}

	__syncthreads();

	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
	{
		odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
	}

}
__global__ void countNonZeroPerRowKernel(int * M, int * M_o, int size)
{	
	__shared__ int smem[BLOCK_WIDTH];
	unsigned int granularity = (size + blockDim.x - 1) / blockDim.x;
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	int tx = threadIdx.x;
	int startIndex = granularity * i;

	smem[tx] = 0;
	__syncthreads();

	//sequential part
	for (int offset = 0; offset < granularity; offset++)
		if (M[i] != 0)
			smem[tx] += 1;
		else
			smem[tx] += 0;

	__syncthreads();
	
		for (unsigned int stride = blockDim.x / 2; stride >= 1; stride >>= 1)
		{
			__syncthreads(); 
			if (tx < stride)
			{
				if (smem[tx + stride] != 0)
					smem[tx] += smem[tx + stride];
				else
					smem[tx] += 0;
			}
		}
		__syncthreads();


		if (tx == 0)
		{
			M_o[blockIdx.x] = smem[0];
		}
}
__global__ void sumReductionKernelNoDivergence(int* A, ull size)
{
	__shared__ int partialSumArr[SHARED_MEM_ARR_SIZE];
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int tx = threadIdx.x;
	if (i < size)
	{
		//each thread loads one el from global memory
		partialSumArr[tx] = A[i];
		for (unsigned int stride = blockDim.x / 2; stride >= 1; stride = stride >> 1)
		{
			__syncthreads();
			if (tx < stride && (i + stride < size))
				partialSumArr[tx] += partialSumArr[tx + stride];
		}
		__syncthreads();

		if (tx == 0)
			A[blockIdx.x] = partialSumArr[0];
	}
}
__global__ void padKernel(int *M, int *Values, int *Columns, int size, int width)
{
	int rowIndex = threadIdx.x + blockDim.x* blockIdx.x;
	int rowStart = rowIndex * size;
	int rowStartELL = rowIndex * width;
	int count = 0;
	for (int n = 0; n < size; n++)
	{
		int el = M[rowStart + n];
		if (el != 0)
		{
			Values[rowStartELL + count] = el;
			Columns[rowStartELL + count] = n;
			count++;
		}
	}
}
__global__ void SpMV_ELL_kernel(int *data, int num_rows,  int *col_indices,  int *x, int *y, int *JDS_Section_width)
{ 
	int outRow = blockIdx.x * blockDim.x + threadIdx.x;
	int numElIdx = blockIdx.x / SECTION_SIZE;
	int sectionIdx = numElIdx * SECTION_SIZE;

	int num_elem;
	if(sectionIdx < num_rows)
	num_elem = JDS_Section_width[sectionIdx];
	if (outRow < num_rows)
	{
		int dot = 0;
		for (int i = 0; i < num_elem; i++)
		{
			//if(data[outRow + i * num_rows] != -1)
				dot += data[outRow + i*num_rows] * x[col_indices[outRow + i * num_rows]];
		}
		y[outRow] += dot;
	}
}

void swap(int* a, int* b);
int partition(int arr[],int* rowArray, int low, int high);
void quickSort(int arr[], int* rowArray, int low, int high);
void SwapRows(int* ValuesOrdered, int* Values, int * col_indices, int * Columns, int* Y, int* Yin, int* JDS_row_index, int maxRowSize, int size);
void SequentialTranspose(int* At, int* A, int width, int height)
{
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			At[i * height + j] = A[j * width + i];
}
void SequentialJDS(int* M, int* X, int* Yseq, int size, int numberOfNonZeros, double sparsity, int* arrOfindices)
{
	srand(time(NULL));
	int* NonZerosCount = new int[size];
	int* JDS_row_index = new int[size];
	int* Y = new int[size];
	for (int i = 0; i < size; i++) JDS_row_index[i] = i;
	memset(NonZerosCount, 0, size*sizeof(int));
	memset(Y, 0, size * sizeof(int));
	 

	for (int i = 0; i < size; i++) X[i] = 1;

	startKer1Seq = clock();
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			if (M[i * size + j] != 0)
				NonZerosCount[i]++;
	endKer1Seq = clock();

#ifdef DEBUG_SEQ

	cout << "Matrix M = " << endl;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			cout << M[i * size + j] << " ";
		}
		cout << " " << NonZerosCount[i] << endl;
	}
#endif
	quickSort(NonZerosCount, JDS_row_index, 0, size);
#ifdef DEBUG_SEQ

	cout << "JDS row indices " << endl;
	for (int i = 0; i < size; i++) cout << JDS_row_index[i] << " ";
	cout << endl;
#endif

	int maxRowSize = NonZerosCount[0];
	int width = maxRowSize, height = size;

#ifdef DEBUG_SEQ
	cout << "Max Number of Zeros in a Row =  " << NonZerosCount[0] << endl;
#endif
	int* Values = new int[width * height];
	int* Columns = new int[width * height];
	int* data = new int[width * height];
	int* col_indices = new int[width * height];
	int meh = width * height;

	for (int i = 0; i < meh; i++) data[i] = 0, Values[i] = 0, Columns[i] = 0, col_indices[i] = 0;
	startKer2Seq = clock();
	int count = 0;
	for (int i = 0; i < size; i++)
	{
		count = 0;
		for (int j = 0; j < size; j++)
		{
			if (M[i * size + j] != 0)
			{
				Values[i*width+count] = M[i * size + j];
				Columns[i*width+count] = j;
				count++;
			}
		}
	}
	endKer2Seq = clock();

#ifdef DEBUG_SEQ

	cout << "Values =  " << endl;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < width; j++)
		{
			cout << Values[i * maxRowSize + j] << " ";
		}
		cout << endl;
	}
#endif

	SwapRows(data, Values, col_indices, Columns,Y, Yseq, JDS_row_index, width, height);
#ifdef DEBUG_SEQ

	cout << "Data =" << endl;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < maxRowSize; j++)
		{
			cout << data[i * maxRowSize + j] << " ";
		}
		cout << endl;
	}
#endif
	startKer4Seq = clock();
	startSeqComp = clock();
	for (int i = 0; i < height; i++)
	{
		int dot = 0;
		for (int j = 0; j < NonZerosCount[i]; j++)
			dot += data[i * width + j] * X[col_indices[i * width + j]];
		Y[i] += dot; 
	}
	endSeqComp = clock();
	endKer4Seq = clock();

	for (int i = 0; i < size; i++) arrOfindices[i] = i;
	sort(arrOfindices, arrOfindices + size, [JDS_row_index](const int& a, const int& b) {return JDS_row_index[a] < JDS_row_index[b]; });

#ifdef DEBUG_SEQ


	cout << "Yseq = " << endl;
	for (int i = 0; i < height; i++)cout << Y[arrOfindices[i]] << " ";
	cout << endl;
#endif

}
int SequentialSameAsParallel(int* Yseq, int* Yparl, int size, int* arrOfindices, int * arrOfindicesParallel)
{
	int count = 0;
	for (int i = 0; i < size; i++)
	{
		if (Yseq[arrOfindices[i]] != Yparl[arrOfindicesParallel[i]])
			count++;
	}
	return count;
}
int main()
{
	srand(time(NULL));
	int size,actualSize, randMeh; double sparsity;
	cout << "Enter size of the matrix N and its sparsity. " << endl;
	cin >> size >> sparsity; 
	//sparsity is ratio of zero to non-zero
	int numberOfNonZeros = (1- sparsity )* size , flatenedSize = size * size;
	actualSize = size;
	if (size % BLOCK_WIDTH != 0)
		size = size + (BLOCK_WIDTH-(size % BLOCK_WIDTH));
	clock_t startSeqFull = clock();
	int* M = new int[size * size];
	int* X = new int[size];
	int* Y = new int[size];
	int* NonZerosCount = new int[size];
	int* JDS_row_index = new int[size];

	for (int i = 0; i < size; i++) JDS_row_index[i] = i;
	memset(NonZerosCount, 0, size);
	memset(M, 0, size*size*sizeof(int));
	memset(Y, 0, size * sizeof(int));

	for (int i = 0; i < size; i++) X[i] = 1;
	//cout << "Main M " << endl;
	for (int i = 0; i < actualSize; i++)
	{
		for (int j = 0; j < actualSize ; j++)
		{
			double probability = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / RAND_BOUND));
			if (probability < (1 - sparsity))
				M[i * size + j] = 1;
		}
	}
	int* Mseq = new int[actualSize * actualSize];
	memset(Mseq, 0, sizeof(int) * actualSize * actualSize);
	for (int i = 0; i < actualSize; i++)
	{
		for (int j = 0; j < actualSize; j++)
		{
			Mseq[i * actualSize + j] = M[i * size + j];
		}
	}

	int* arrOfindices = new int[size];
	int* arrOfindicesParallel = new int[size];
#ifdef TEST
	M[0] = 3;
	M[1] = 0;
	M[2] = 1;
	M[3] = 0;
	M[30] = 10;
	
	M[1024] = 0;
	M[1025] = 0;
	M[1026] = 0;
	M[1027] = 0;

	M[2048] = 0;
	M[2049] = 2;
	M[2050] = 4;
	M[2051] = 1;

	M[3072] = 1;
	M[3073] = 0;
	M[3074] = 0;
	M[3075] = 1;

	M[5120] = 10;
	M[5121] = 10;
	M[5122] = 10;
	M[5173] = 10;
	M[5174] = 10;
	M[5175] = 10;
	M[5176] = 10;
#endif
	clock_t startSeqAfterMem = clock();
	int* Yseq = new int[size];
	memset(Yseq, 0, size * sizeof(int));
	SequentialJDS(Mseq, X, Yseq, actualSize, numberOfNonZeros, sparsity, arrOfindices);
	endSeqFull  = clock();

	startGPUFull = clock();
	cudaError_t cudaStatus = TheWrapperofWrappers(NonZerosCount, M, size, JDS_row_index, X, Y, actualSize, arrOfindicesParallel);
	endGPUFull = clock();
	cout << "Number of differences between parallel Vs. sequential= " << SequentialSameAsParallel(Yseq, Y, actualSize, arrOfindices, arrOfindicesParallel) << endl << endl;
	//Kernel IDs:
	//1-> count non-zero(sum reduction).2-> padKernel.	3->transpose.	 4->SpMV
	std::cout << std::setprecision(8) << std::fixed;
	cout << "1) Non-zeros Count Step: \nGPU Time= " << std::setprecision(6) << double(double(endKer1 - startKer1) / (double)CLOCKS_PER_SEC) << "| CPU Time= " << double(endKer1Seq - startKer1Seq) / CLOCKS_PER_SEC << setw(3) << "| Speedup= " << (double)(endKer1Seq - startKer1Seq) / (endKer1 - startKer1) << endl << endl;
	cout << "2) Creating Values and Column matrices (Padding):\nGPU Time= " << std::setprecision(6) << double(endKer2 - startKer2) / CLOCKS_PER_SEC << "| CPU Time= " << double(endKer2Seq - startKer2Seq) / CLOCKS_PER_SEC << setw(3) << "| Speedup= " << (double)(endKer2Seq - startKer2Seq) / (endKer2 - startKer2) << endl << endl;
	cout << "3) Transpose: \nGPU Time= " << std::setprecision(6) << double(endKer3 - startKer3) / CLOCKS_PER_SEC << "| CPU= N/A. " << endl << endl;
	cout << "4) SpMV Calculation: \nGPU Time= " << std::setprecision(6) << double(endKer4 - startKer4) / CLOCKS_PER_SEC << "| CPU Time= " << double(endKer4Seq - startKer4Seq) / CLOCKS_PER_SEC << setw(3) << "| Speedup= " << (double)(endKer4Seq - startKer4Seq) / (endKer4 - startKer4) << endl << endl;

	//Seuquential
	cout << "Full time:\nGPU= " << double(endGPUFull - startGPUFull) / CLOCKS_PER_SEC;  cout << "| CPU Time= " << (double)(endSeqFull - startSeqFull) / CLOCKS_PER_SEC << "| Speedup= " << (double)(endSeqFull - startSeqFull) / (endGPUFull - startGPUFull) << endl << endl;
	cout << "SpMV Computation:\nGPU= " << double(abs(endGPUcomp - startGPUcomp)) / CLOCKS_PER_SEC;  cout << "| CPU Time= " << (double)(endSeqComp - startSeqComp) / CLOCKS_PER_SEC; cout << setw(3) << "| Speedup= " << (double)(endSeqComp - startSeqComp) / (endGPUcomp - startGPUcomp) << endl << endl;
	double KernelsTime = (double)((endKer1-startKer1) + (endKer2-startKer2) + 2*(endKer3-startKer3) + (endKer4-startKer4)) / CLOCKS_PER_SEC;
	cout << "GPU All Kernels without Memory= " << KernelsTime << "| CPU Time= " << (double)(endSeqFull - startKer1Seq) / CLOCKS_PER_SEC; cout << "| Speedup= " << (double)(endSeqFull - startKer1Seq) / CLOCKS_PER_SEC / KernelsTime << endl << endl;
	cout << "GPU All Kernels with Memory= " << double(endGPUFull-startGPUFull)/ CLOCKS_PER_SEC << "| CPU Time= " << (double)(endSeqFull - startSeqAfterMem) / CLOCKS_PER_SEC; cout << "| Speedup= " << (double)(endSeqFull - startSeqAfterMem) / (endGPUFull - startGPUFull)<< endl << endl;

	free(M);
	free(X);
	free(Y);
	free(Yseq);
	free(NonZerosCount);
	free(JDS_row_index);
	free(arrOfindicesParallel);
	free(arrOfindices);

    return 0;
}

cudaError_t TheWrapperofWrappers(int* NonZerosCount,  int* M, unsigned int size, int* JDS_row_index, int* X, int* Yin, int actualSize, int* arrOfindices)
{
	int* dev_M = 0;
	int* dev_Mo = 0;
	cudaError_t cudaStatus;
#ifdef parallelDEBUG
	cout << "Parallel M = " << endl;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			cout << M[i * size + j] << " ";
		}
		cout << endl;
	}
#endif

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 	{ fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");  goto Error; }
    cudaStatus = cudaMalloc((void**)&dev_M, size *size* sizeof(int));
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMalloc((void**)&dev_Mo, size * sizeof(int));
	if (cudaStatus != cudaSuccess) 	{fprintf(stderr, "cudaMalloc failed!");	goto Error;	}
    cudaStatus = cudaMemcpy(dev_M, M, size * size* sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!");   goto Error; }

	dim3 dimBlock(BLOCK_WIDTH, 1, 1);
	//Which is always adjusted to be a multiple of BLOCK_WIDTH no matter what the user's input is
	dim3 dimGrid(size, 1, 1);

	startKer1 = clock();
	countNonZeroPerRowKernel <<<dimGrid, dimBlock >>>(dev_M, dev_Mo, size);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }
    cudaStatus = cudaDeviceSynchronize();
	endKer1 = clock();

    if (cudaStatus != cudaSuccess) 	{ fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);  goto Error;}
    cudaStatus = cudaMemcpy(NonZerosCount, dev_Mo, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!");  goto Error; }

	//I can still use dev_M since it is not overwritten anywhere without reallocation

	quickSort(NonZerosCount, JDS_row_index, 0, size);
#ifdef parallelDEBUG

	cout << "Non zeroz count " << endl;
	for (int i = 0; i < actualSize; i++)
		cout << NonZerosCount[i] << " ";
	cout << endl;
#endif
	int maxRowSize = NonZerosCount[0];
	int width = maxRowSize, height = size;
	//Make width and height always an integer multiple of TILE_DIM
	if (maxRowSize % TILE_DIM != 0)
		width = TILE_DIM-(maxRowSize % TILE_DIM) + maxRowSize;
	if (size % TILE_DIM != 0)
		height = TILE_DIM-(size % TILE_DIM) + size;
	int* Values = new int[width * height];
	int* Columns = new int[width * height];
	int* data = new int[width * height];
	int* col_indices = new int[width * height];
	int* Y = new int[height];
	int meh = width * height;
	for (int i = 0; i < meh; i++) data[i] = 0, Values[i] =0, Columns[i] = 0, col_indices[i] = 0;
	cudaStatus = toELL(dev_M, Values, Columns, width, height, maxRowSize);
	//Reorder rows to generate Jagged Diagonal Matrix in data matrix
#ifdef	parallelDEBUG

	cout << "Values " << endl;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			cout << Values[i * width + j] << " ";
		}
		cout << endl;
	}
#endif
	SwapRows(data, Values, col_indices, Columns, Y, Yin, JDS_row_index, width, height);
	//Reorder rows to generate Jagged Diagonal Matrix in data matrix
#ifdef	parallelDEBUG

	cout << "Values after swap rows" << endl;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			cout << data[i * width + j] << " ";
		}
		cout << endl;
	}
#endif

	//Get transpose of data and col_indices consequently
	cudaStatus = transpose(data, height, width);
	cudaStatus = transpose(col_indices, height, width);
#ifdef	parallelDEBUG
	cout << "data transpose " << endl;
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			cout << data[i*height+j] << " ";
		}
		cout << endl;
	}

	cout << endl;
	cout << "parallel col_indices " << endl;

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			cout << col_indices[i * height + j] << " ";
		}
		cout << endl;
	}
#endif 
	//Multiplication Kernel
	startGPUcomp = clock();
	cudaStatus = spMV_Sectioned_ELL(data, col_indices, NonZerosCount, JDS_row_index, X, Y, maxRowSize, size);
	endGPUcomp = clock();



	for (int i = 0; i < size; i++) arrOfindices[i] = i;
	sort(arrOfindices, arrOfindices + size, [JDS_row_index](const int& a, const int& b) {return JDS_row_index[a] < JDS_row_index[b]; });

#ifdef parallelDEBUG
	cout << "JDS row index " << endl;
	for (int i = 0; i < size; i++)
		cout << JDS_row_index[i] << " ";
	cout << endl;
	//Access sorted Y (result vector)
	int act = actualSize;
	cout << "Y parallel = " << endl;
	for (int i = 0; i < act; i++)cout << Y[arrOfindices[i]] << " ";
	cout << endl;
#endif // !DEBUG



Error:
    cudaFree(dev_Mo);
    cudaFree(dev_M);
	free(Values);
	free(Columns);
	free(data);
	free(col_indices);
    
    return cudaStatus;
}
cudaError_t toELL( int *dev_M, int* Values, int* Columns, int width, int size, int maxZeroCount)
{
	int* dev_Values = 0;
	int* dev_Columns = 0;
	cudaError_t cudaStatus;
	//maxRow size = width. size = height. maxZeroCount is actual width
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}


	cudaStatus = cudaMalloc((void**)&dev_Values, width * size*sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_Columns, width * size * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(dev_Values, Values, size * width * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_Columns, Columns, size * width * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//1 block of 1024 to handle 1024 rows.
	dim3 dimBlock(BLOCK_WIDTH, 1, 1);
	dim3 dimGrid(ceil((float)size/BLOCK_WIDTH), 1, 1);

	startKer2 = clock();
	padKernel << <dimGrid, dimBlock >> > (dev_M, dev_Values, dev_Columns, size, width);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	endKer2 = clock();

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(Values, dev_Values, width * size* sizeof(int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(Columns, dev_Columns, width * size* sizeof(int), cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_Values);
	cudaFree(dev_Columns);

	return cudaStatus;
}
cudaError_t spMV_Sectioned_ELL(int* data, int* col_indices, int* JDS_section_width, int* JDS_row_index, int *x, int *y, int  maxRowSize, int size)
{
	int* dev_data = 0;
	int* dev_col_indices = 0;
	int* dev_JDS_section_width = 0;
	int* dev_x = 0;
	int* dev_y = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)	{fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");goto Error;}

	cudaStatus = cudaMalloc((void**)&dev_data, maxRowSize * size * sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_x, size * sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_y, size * sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_col_indices, maxRowSize * size * sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_JDS_section_width,  size * sizeof(int));

	if (cudaStatus != cudaSuccess){	fprintf(stderr, "cudaMalloc failed!");	goto Error;	}

	cudaStatus = cudaMemcpy(dev_data, data, maxRowSize * size * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_col_indices, col_indices, maxRowSize * size * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_JDS_section_width, JDS_section_width, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_x, x, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_y, y, size * sizeof(int), cudaMemcpyHostToDevice);

	if (cudaStatus != cudaSuccess){	fprintf(stderr, "cudaMemcpy failed!");goto Error;}

	//A block handle BLOCK_WIDTH elements in the output vector
	//size is already modified to be an integer multiple of BLOCK_WIDTH
	dim3 dimBlock(BLOCK_WIDTH, 1, 1);
	dim3 dimGrid(ceil((float)size  /BLOCK_WIDTH), 1, 1);
	startGPUcomp = clock();
	startKer4 = clock();
	SpMV_ELL_kernel << <dimGrid, dimBlock >> > (dev_data, size, dev_col_indices, dev_x, dev_y, dev_JDS_section_width);


	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess){	fprintf(stderr, "SpME kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));	goto Error;}
	cudaStatus = cudaDeviceSynchronize();
	endKer4 = clock();
	endGPUcomp = clock();
	if (cudaStatus != cudaSuccess){	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching SpME!\n", cudaStatus);goto Error;	}
	cudaStatus = cudaMemcpy(y, dev_y,  size * sizeof(int), cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!");goto Error;}

Error:
	cudaFree(dev_col_indices);
	cudaFree(dev_data);
	cudaFree(dev_JDS_section_width);
	cudaFree(dev_x);
	cudaFree(dev_y);

	return cudaStatus;
}
cudaError_t transpose(int* A,  int height, int width)
{
	int* dev_A = 0;
	int* dev_Ao = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)	{fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");goto Error;}
	cudaStatus = cudaMalloc((void**)&dev_A, width * height * sizeof(int));
	if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");goto Error;}
	cudaStatus = cudaMalloc((void**)&dev_Ao, width * height * sizeof(int));
	if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!");	goto Error;}
	cudaStatus = cudaMemcpy(dev_A, A, height * width *sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess){	fprintf(stderr, "cudaMemcpy failed!");	goto Error;}


	dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
	dim3 dimGrid(ceil((float)width /TILE_DIM), ceil((float)height / BLOCK_ROWS), 1);

	startKer3 = clock();
	transposeNoBankConflicts << < dimGrid, dimBlock >> > (dev_Ao, dev_A, width, height);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess){	fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));goto Error;}
	cudaStatus = cudaDeviceSynchronize();
	endKer3 = clock();

	if (cudaStatus != cudaSuccess){fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);goto Error;}
	cudaStatus = cudaMemcpy(A, dev_Ao, height * width *sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess){	fprintf(stderr, "cudaMemcpy failed!");goto Error;}

Error:
	cudaFree(dev_A);
	cudaFree(dev_Ao);
	return cudaStatus;
}
void SwapRows(int* ValuesOrdered, int* Values, int* col_indices, int* Columns, int* Y, int* Yin, int* JDS_row_index, int maxRowSize, int size)
{
	for (int rows = 0; rows < size; rows++)
	{
		for (int cols = 0; cols < maxRowSize; cols++)
		{
			ValuesOrdered[rows * maxRowSize + cols] = Values[JDS_row_index[rows] * maxRowSize + cols];
			col_indices[rows * maxRowSize + cols] = Columns[JDS_row_index[rows] * maxRowSize + cols];
		}
		Y[rows ] = Yin[JDS_row_index[rows]];

	}
}
void swap(int* a, int* b)
{
	int t = *a;
	*a = *b;
	*b = t;
}
int partition(int arr[], int* rowArray, int low, int high)
{
	int pivot = arr[high];    // pivot 
	int i = (low - 1);  // Index of smaller element 

	for (int j = low; j <= high - 1; j++)
	{
		if (arr[j] >= pivot)
		{
			i++;
			swap(&arr[i], &arr[j]);
			swap(&rowArray[i], &rowArray[j]);
		}
	}
	swap(&arr[i + 1], &arr[high]);
	swap(&rowArray[i + 1], &rowArray[high]);

	return (i + 1);
}
void quickSort(int arr[], int* rowArray, int low, int high)
{
	if (low < high)
	{
		int pi = partition(arr, rowArray, low, high);
		quickSort(arr, rowArray, low, pi - 1);
		quickSort(arr, rowArray, pi + 1, high);
	}
}
