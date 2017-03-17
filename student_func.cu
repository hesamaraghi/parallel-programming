/* Udacity Homework 3
HDR Tone-mapping

Background HDR
==============

A High Dynamic Range (HDR) image contains a wider variation of intensity
and color than is allowed by the RGB format with 1 byte per channel that we
have used in the previous assignment.

To store this extra information we use single precision floating point for
each channel.  This allows for an extremely wide range of intensity values.

In the image for this assignment, the inside of church with light coming in
through stained glass windows, the raw input floating point values for the
channels range from 0 to 275.  But the mean is .41 and 98% of the values are
less than 3!  This means that certain areas (the windows) are extremely bright
compared to everywhere else.  If we linearly map this [0-275] range into the
[0-255] range that we have been using then most values will be mapped to zero!
The only thing we will be able to see are the very brightest areas - the
windows - everything else will appear pitch black.

The problem is that although we have cameras capable of recording the wide
range of intensity that exists in the real world our monitors are not capable
of displaying them.  Our eyes are also quite capable of observing a much wider
range of intensities than our image formats / monitors are capable of
displaying.

Tone-mapping is a process that transforms the intensities in the image so that
the brightest values aren't nearly so far away from the mean.  That way when
we transform the values into [0-255] we can actually see the entire image.
There are many ways to perform this process and it is as much an art as a
science - there is no single "right" answer.  In this homework we will
implement one possible technique.

Background Chrominance-Luminance
================================

The RGB space that we have been using to represent images can be thought of as
one possible set of axes spanning a three dimensional space of color.  We
sometimes choose other axes to represent this space because they make certain
operations more convenient.

Another possible way of representing a color image is to separate the color
information (chromaticity) from the brightness information.  There are
multiple different methods for doing this - a common one during the analog
television days was known as Chrominance-Luminance or YUV.

We choose to represent the image in this way so that we can remap only the
intensity channel and then recombine the new intensity values with the color
information to form the final image.

Old TV signals used to be transmitted in this way so that black & white
televisions could display the luminance channel while color televisions would
display all three of the channels.


Tone-mapping
============

In this assignment we are going to transform the luminance channel (actually
the log of the luminance, but this is unimportant for the parts of the
algorithm that you will be implementing) by compressing its range to [0, 1].
To do this we need the cumulative distribution of the luminance values.

Example
-------

input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
min / max / range: 0 / 9 / 9

histo with 3 bins: [4 7 3]

cdf : [4 11 14]


Your task is to calculate this cumulative distribution by following these
steps.

*/

#include "utils.h"
#include <stdio.h>
#include <thrust/extrema.h>
// #include "reference_calc.h"
#define maxThreadsPerBlock  1024
#define itemsPerThread  16
#define binsPerThread  8
#define blockSize_x  64
#define blockSize_y  (maxThreadsPerBlock/blockSize_x)
#define bitMaxThreadsPerBlock 10
#define MAX_BUFF_COUNT 10



__global__ void shmem_reduce_min_kernel(float * d_out, const float * d_in, const int size)
{
	// sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
	extern __shared__ float sdata[];

	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	unsigned int s = blockDim.x / 2;
	// load shared mem from global mem
	if (tid < s)
	{
		sdata[tid] = min(d_in[(myId < size) ? myId : (size - 1)], d_in[((myId + s) < size) ? (myId + s) : (size - 1)]);
	}
	else
	{
		return;
	}
	__syncthreads();            // make sure entire block is loaded!

	// do reduction in shared mem
	for (s >>= 1; s > 1; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = min(sdata[tid], sdata[tid + s]);
		}
		__syncthreads();        // make sure all adds at one stage are done!
	}

	// only thread 0 writes result for this block back to global mem
	if (tid == 0)
	{
		d_out[blockIdx.x] = min(sdata[0], sdata[1]);
	}
}

__global__ void shmem_reduce_max_kernel(float * d_out, const float * d_in, const int size)
{
	// sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
	extern __shared__ float sdata[];

	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	unsigned int s = blockDim.x / 2;
	// load shared mem from global mem
	if (tid < s)
	{
		sdata[tid] = max(d_in[(myId < size) ? myId : (size - 1)],
			d_in[((myId + s) < size) ? (myId + s) : (size - 1)]);
	}
	else
	{
		return;
	}
	__syncthreads();            // make sure entire block is loaded!

	// do reduction in shared mem
	for (s >>= 1; s > 1; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = max(sdata[tid], sdata[tid + s]);
		}
		__syncthreads();        // make sure all adds at one stage are done!
	}

	// only thread 0 writes result for this block back to global mem
	if (tid == 0)
	{
		d_out[blockIdx.x] = max(sdata[0], sdata[1]);
	}
}

void min_max_finding(float * min_logLum,
	float * max_logLum,
	const float* const d_in,
	int size)
{
	// assumes that size is not greater than maxThreadsPerBlock^2
	// and that size is a multiple of maxThreadsPerBlock

	float * d_out_min, *d_out_max;


	checkCudaErrors(cudaMalloc((void **)&d_out_min, sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_out_max, sizeof(float)));


	int blocks = (size / maxThreadsPerBlock) + ((size % maxThreadsPerBlock == 0) ? 0 : 1);
	int size_ = size;
	float *d_intermediate;
	float *d_intermediate2;
	if (blocks == 1)
	{
		shmem_reduce_min_kernel <<<1, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float) / 2 >>>
			(d_out_min, d_in, size_);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}
	else
	{
		checkCudaErrors(cudaMalloc(&d_intermediate, (size_t)blocks * sizeof(float)));
		shmem_reduce_min_kernel <<<blocks, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float) / 2 >>>
			(d_intermediate, d_in, size_);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		size_ = blocks;
		blocks = (blocks >> bitMaxThreadsPerBlock) + 1;
		if (blocks == 1)
		{
			shmem_reduce_min_kernel <<<1, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float) / 2 >>>
				(d_out_min, d_intermediate, size_);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		}
		else
		{
			checkCudaErrors(cudaMalloc(&d_intermediate2, (size_t)blocks * sizeof(float)));
			shmem_reduce_min_kernel <<<blocks, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float) / 2 >>>
				(d_intermediate2, d_intermediate, size_);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			shmem_reduce_min_kernel <<<1, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float) / 2 >>>
				(d_out_min, d_intermediate2, blocks);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		}
	}

	blocks = (size >> bitMaxThreadsPerBlock) + 1;
	if (blocks == 1)
	{
		shmem_reduce_max_kernel <<<1, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float) / 2 >>>
			(d_out_max, d_in, size);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}
	else
	{
		shmem_reduce_max_kernel <<<blocks, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float) / 2 >>>
			(d_intermediate, d_in, size);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		size = blocks;
		blocks = (blocks >> bitMaxThreadsPerBlock) + 1;
		if (blocks == 1)
		{
			shmem_reduce_max_kernel <<<1, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float) / 2 >>>
				(d_out_max, d_intermediate, size);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaFree(d_intermediate));
		}
		else
		{
			shmem_reduce_max_kernel <<<blocks, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float) / 2 >>>
				(d_intermediate2, d_intermediate, size);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			shmem_reduce_max_kernel <<<1, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float) / 2 >>>
				(d_out_max, d_intermediate2, blocks);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaFree(d_intermediate));
			checkCudaErrors(cudaFree(d_intermediate2));
		}
	}

	checkCudaErrors(cudaMemcpy(min_logLum, d_out_min, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(max_logLum, d_out_max, sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_out_min));
	checkCudaErrors(cudaFree(d_out_max));
}

__global__ void total_atomic_histo(int *d_bins, const float *d_in, const float min_logLum,
	const float max_logLum, const int numBins, const int size)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	if (myId >= size) return;
	unsigned int bin = min(static_cast<unsigned int>(numBins - 1),
		static_cast<unsigned int>((d_in[myId] - min_logLum) / (max_logLum - min_logLum) * numBins));
	atomicAdd(&(d_bins[bin]), 1);
}

__global__ void reduce_atomic_histo(int *d_bins, const float *d_in, const float min_logLum,
	const float max_logLum, const int numBins, const int size)
{
	extern __shared__ int sdataint[];
	int myIdx_x = threadIdx.x + blockDim.x * blockIdx.x;
	int myIdx_y = threadIdx.y + blockDim.y * blockIdx.y;
	int i, tid = threadIdx.x + blockDim.x * threadIdx.y;
	int bin, s;
	int itemIdx0 = myIdx_x * itemsPerThread,
		binIdx0 = myIdx_y * binsPerThread;
	int localHisto[binsPerThread];

	for (i = 0; i < binsPerThread; i++) localHisto[i] = 0;

	if (itemIdx0 < size)
	{
		for (i = 0; itemIdx0 + i < size && i < itemsPerThread; i++)
		{
			bin = static_cast<unsigned int>((d_in[itemIdx0 + i] - min_logLum)
				/ (max_logLum - min_logLum) * numBins);
			bin = (bin == numBins) ? bin - 1 : bin;
			if (bin >= binIdx0 && bin < (binIdx0 + binsPerThread))
				localHisto[bin % binsPerThread] = localHisto[bin % binsPerThread] + 1;
		}
	}


	for (i = 0; i < binsPerThread; i++)
	{
		s = blockDim.x;
		sdataint[tid] = localHisto[i];
		__syncthreads();
		for (s >>= 1; s > 0; s >>= 1)
		{
			if (threadIdx.x < s)
			{
				sdataint[tid] = sdataint[tid] + sdataint[tid + s];
			}
			__syncthreads();
		}
		if (threadIdx.x == 0 && binIdx0 + i < numBins) atomicAdd(&(d_bins[binIdx0 + i]), sdataint[blockDim.x * threadIdx.y]);
		__syncthreads();
	}
}

__global__ void histogram_gmem_atomics(int *out, const float *d_in, const float min_logLum,
	const float max_logLum, const int numBins, const int numRows, const int numCols)
{
	// pixel coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// grid dimensions
	int nx = blockDim.x * gridDim.x;
	int ny = blockDim.y * gridDim.y;

	// linear thread index within 2D block
	int t = threadIdx.x + threadIdx.y * blockDim.x;

	// total threads in 2D block
	int nt = blockDim.x * blockDim.y;

	// linear block index within 2D grid
	int g = blockIdx.x + blockIdx.y * gridDim.x;

	// initialize temporary accumulation array in global memory
	int *gmem = out + g * numBins;
	for (int i = t; i < numBins; i += nt) gmem[i] = 0;

	// process pixels
	// updates our block's partial histogram in global memory
	for (int col = x; col < numCols; col += nx)
	for (int row = y; row < numRows; row += ny) {
		unsigned int bin = min(static_cast<unsigned int>(numBins - 1),
			static_cast<unsigned int>((d_in[row * numCols + col] - min_logLum) /
			(max_logLum - min_logLum) * numBins));
		atomicAdd(&gmem[bin], 1);
	}
}

__global__ void histogram_final_accum(const int *in, int n, int *d_bins, const int numBins)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numBins) {
		unsigned int total = 0;
		for (int j = 0; j < n; j++)
			total += in[i + numBins * j];
		d_bins[i] = total;
	}
}

__global__ void histogram_final_accum_reduce(const int *in, int n, int *d_bins, const int numBins)
{
	extern __shared__ int sdata_[];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// grid dimensions
	int nx = blockDim.x * gridDim.x;
	int ny = blockDim.y * gridDim.y;

	// linear thread index within 2D block
	int tid = threadIdx.x;
	int s;
	for (int row = y; row < numBins; row += ny) {
		sdata_[tid] = 0;
		for (int col = x; col < n; col += nx)
			sdata_[tid] = in[col * numBins + row] + sdata_[tid];
		__syncthreads();
		s = maxThreadsPerBlock;
		for (s >>= 1; s > 0; s >>= 1)
		{
			if (tid < s)
				sdata_[tid] = sdata_[tid] + sdata_[tid + s];
			__syncthreads();
		}
		if (tid == 0) atomicAdd(&d_bins[row], sdata_[tid]);
	}
}

__global__ void histogram_smem_atomics(int *out, const float *d_in, const float min_logLum,
	const float max_logLum, const int numBins, const int numRows, const int numCols)
{
	// pixel coordinates
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// grid dimensions
	int nx = blockDim.x * gridDim.x;
	int ny = blockDim.y * gridDim.y;

	// linear thread index within 2D block
	int t = threadIdx.x + threadIdx.y * blockDim.x;

	// total threads in 2D block
	int nt = blockDim.x * blockDim.y;

	// linear block index within 2D grid
	int g = blockIdx.x + blockIdx.y * gridDim.x;

	// initialize temporary accumulation array in shared memory
	extern __shared__ int smem[];
	for (int i = t; i < numBins; i += nt) smem[i] = 0;
	__syncthreads();

	// process pixels
	// updates our block's partial histogram in shared memory
	for (int col = x; col < numCols; col += nx)
	for (int row = y; row < numRows; row += ny) {
		unsigned int bin = min(static_cast<unsigned int>(numBins - 1),
			static_cast<unsigned int>((d_in[row * numCols + col] - min_logLum) /
			(max_logLum - min_logLum) * numBins));
		atomicAdd(&smem[bin], 1);
	}
	__syncthreads();

	// write partial histogram into the global memory
	out += g * numBins;
	for (int i = t; i < numBins; i += nt) {
		out[i] = smem[i];
	}
}

__global__ void scanInclusiveShared(
	int *d_Dst,
	int *d_Src,
	int size)
{
	__shared__ int s_Data[2 * maxThreadsPerBlock];

	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	int temp;

	s_Data[tid] = 0;
	tid += maxThreadsPerBlock;
	s_Data[tid] = (pos < size) ? d_Src[pos] : 0;

	for (int offset = 1; offset < maxThreadsPerBlock; offset <<= 1)
	{
		__syncthreads();
		temp = s_Data[tid] + s_Data[tid - offset];
		__syncthreads();
		s_Data[tid] = temp;
	}

	if (pos < size) d_Dst[pos] = s_Data[tid];
}

__global__ void reducedScanInclusiveShared(
	int *d_Dst,
	int *d_Src,
	int size,
	int size_big)
{
	__shared__ int s_Data[2 * maxThreadsPerBlock];

	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	int temp;

	s_Data[tid] = 0;
	tid += maxThreadsPerBlock;
	temp = pos * maxThreadsPerBlock + maxThreadsPerBlock - 1;
	s_Data[tid] = (pos < size) ? (temp < size_big ? d_Src[temp] : d_Src[size_big - 1]) : 0;

	for (int offset = 1; offset < maxThreadsPerBlock; offset <<= 1)
	{
		__syncthreads();
		temp = s_Data[tid] + s_Data[tid - offset];
		__syncthreads();
		s_Data[tid] = temp;
	}

	if (pos < size) d_Dst[pos] = s_Data[tid];
}

__global__ void inclusiveToExclusive(
	int *d_Dst,
	int size
	)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int temp;
	if (pos < size - 1){
		temp = d_Dst[pos];
		__syncthreads();
		d_Dst[pos + 1] = temp;
	}
	if (pos == 0) d_Dst[0] = 0;
}

__global__ void backwardAdder(
	int *d_Dst,
	int *d_Src,
	int size)
{
	__shared__ int s_Data;

	int pos_dst = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	if (tid == 0 && blockIdx.x > 0) s_Data = d_Src[blockIdx.x - 1];
	__syncthreads();
	if (pos_dst < size && blockIdx.x > 0) d_Dst[pos_dst] += s_Data;
}

inline void iterativeBuffer(int *d_Dst,
	int *d_Src,
	int size)
{
	int** d_Buf;
	d_Buf = (int**)malloc(MAX_BUFF_COUNT * sizeof(int*));
	int block_size[MAX_BUFF_COUNT];
	int itr_size[MAX_BUFF_COUNT];
	block_size[0] = (size + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
	itr_size[0] = size;
	d_Buf[0] = d_Dst;
	scanInclusiveShared <<<block_size[0], maxThreadsPerBlock >>>(d_Buf[0], d_Src, size);

	int i;
	for (i = 1 ; block_size[i - 1] > 1; ++i) {
		itr_size[i] = block_size[i - 1];
		block_size[i] = (block_size[i - 1] + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
		checkCudaErrors(cudaMalloc((void **)&d_Buf[i], sizeof(int)* itr_size[i]));
		reducedScanInclusiveShared<<<block_size[i], maxThreadsPerBlock >>>(d_Buf[i], d_Buf[i - 1], itr_size[i], itr_size[i - 1]);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}

	for (--i; i > 0; --i) {
		backwardAdder <<<block_size[i - 1], maxThreadsPerBlock >>>(d_Buf[i - 1], d_Buf[i], itr_size[i - 1]);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaFree(d_Buf[i]));
	}
	inclusiveToExclusive <<<block_size[0], maxThreadsPerBlock >>>(d_Buf[0],size);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	free(d_Buf);
}






void your_histogram_and_prefixsum(const float* const d_logLuminance,
	unsigned int* const d_cdf,
	float &min_logLum,
	float &max_logLum,
	const size_t numRows,
	const size_t numCols,
	const size_t numBins)
{

	int size = numRows * numCols;

	min_max_finding(&min_logLum, &max_logLum, d_logLuminance, size);

	int *d_bins;

	checkCudaErrors(cudaMalloc((void **)&d_bins, sizeof(int)* numBins));
	checkCudaErrors(cudaMemset(d_bins, 0, sizeof(int)* numBins));

	// totally done by atomic add

	// int blocks = (size / maxThreadsPerBlock) + ((size % maxThreadsPerBlock == 0) ? 0 : 1);

	// total_atomic_histo<<<blocks,maxThreadsPerBlock>>>(d_bins, d_logLuminance, min_logLum,
	//                    max_logLum, numBins, size);
	// cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// first reduce then atomic add

	// int numBlocks_x = (size / (blockSize_x * itemsPerThread)) + ((size % (blockSize_x * itemsPerThread) == 0) ? 0 : 1);
	// int numBlocks_y = (numBins / (blockSize_y * binsPerThread)) + ((numBins % (blockSize_y * binsPerThread) == 0) ? 0 : 1);
	// dim3 grid(numBlocks_x, numBlocks_y);
	// dim3 block(blockSize_x, blockSize_y);

	// reduce_atomic_histo<<<grid, block, maxThreadsPerBlock * sizeof(int)>>>(d_bins, d_logLuminance, min_logLum,
	//                    max_logLum, numBins, size);
	// cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// first add local histograms to the global memory
	int numBlocks_x = numRows / blockSize_x;
	int numBlocks_y = numCols / blockSize_y;
	dim3 grid(numBlocks_x, numBlocks_y);
	dim3 block(blockSize_x, blockSize_y);
	int *d_gmem_bins;
	checkCudaErrors(cudaMalloc((void **)&d_gmem_bins, sizeof(int)* numBins * numBlocks_x * numBlocks_y));

	// histogram_gmem_atomics<<<grid, block>>>(d_gmem_bins, d_logLuminance, min_logLum,
	//                                        max_logLum, numBins, numRows, numCols);
	// cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	histogram_smem_atomics <<<grid, block, numBins * sizeof(int) >>>(d_gmem_bins, d_logLuminance, min_logLum,
		max_logLum, numBins, numRows, numCols);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// int Blocks = numBins / maxThreadsPerBlock + ((numBins % maxThreadsPerBlock == 0) ? 0 : 1);

	// histogram_final_accum<<<Blocks,maxThreadsPerBlock>>>(d_gmem_bins, numBlocks_x * numBlocks_y, d_bins, numBins);
	// cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	int grid_x = (numBlocks_x * numBlocks_y) / maxThreadsPerBlock;
	int grid_y = numBins;
	dim3 grid_((grid_x == 0) ? 1 : grid_x, grid_y);
	dim3 block_(maxThreadsPerBlock, 1);

	histogram_final_accum_reduce <<<grid_, block_, sizeof(int)* maxThreadsPerBlock >>>(d_gmem_bins, numBlocks_x * numBlocks_y, d_bins, numBins);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaFree(d_gmem_bins));

	iterativeBuffer((int*) d_cdf, d_bins, numBins);

	checkCudaErrors(cudaFree(d_bins));


	//TODO
	/*Here are the steps you need to implement
	1) find the minimum and maximum value in the input logLuminance channel
	store in min_logLum and max_logLum
	2) subtract them to find the range
	3) generate a histogram of all the values in the logLuminance channel using
	the formula: bin = (lum[i] - lumMin) / lumRange * numBins
	4) Perform an exclusive scan (prefix sum) on the histogram to get
	the cumulative distribution of luminance values (this should go in the
	incoming d_cdf pointer which already has been allocated for you)       */


}
