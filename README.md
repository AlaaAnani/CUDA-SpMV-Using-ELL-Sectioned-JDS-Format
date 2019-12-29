# SpMV-Using-ELL-Sectioned-JDS-Format
A C++ with CUDA C program that generates a random square matrix A of any size NxN (It doesn't have to be a power of 2 sine it is always padded to be an integer multiple of the block width!) with mostly zeros, and two vectors X and Y of size N. 
The sparsity of the matrix (i.e., the ratio of zero to non-zero elements) should be specified by the user. 
My program converts the matrix A to the ELL-sectioned JDS format and then computes the SpMV operation on it. 
The conversion itself is parallelized.

Kernel IDs:
1-> count non-zeros per row(sum reduction).
2-> padKernel.	
3->transpose.	 
4->SpMV

This program also calculates the performance of the following:
1. The kernel calls times independetly. There are 4 kernels in the conversion and calculation process.
2. The sequential times for each step equivalent to he 4 kernels.
3. The entire kernel times (All 4 combined).
4. Time of the parallel code including memory allocation to the device (The time around the wrapper of all wrappers).
