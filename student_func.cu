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
#define bitMaxThreadsPerBlock 10


__global__ void shmem_reduce_min_kernel(float * d_out, const float * d_in,const int size)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;
    unsigned int s = blockDim.x / 2;
    // load shared mem from global mem
    if (tid < s) 
    {
      sdata[tid] = min(d_in[(myId < size) ? myId : (size - 1)],d_in[((myId + s) < size) ? (myId + s) : (size - 1)]);
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

__global__ void shmem_reduce_max_kernel(float * d_out, const float * d_in,const int size)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;
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

void min_max_finding(float * d_out_min, 
                     float * d_out_max, 
                     const float* const d_in, 
                     int size)
{
    // assumes that size is not greater than maxThreadsPerBlock^2
    // and that size is a multiple of maxThreadsPerBlock

    int blocks = (size / maxThreadsPerBlock) + ((size % maxThreadsPerBlock == 0) ? 0 : 1);
    int size_ = size;
    float *d_intermediate;
    float *d_intermediate2;
    if(blocks == 1)
    {
      shmem_reduce_min_kernel<<<1, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float) / 2>>>
        (d_out_min, d_in, size_);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }
    else
    {
      checkCudaErrors(cudaMalloc(&d_intermediate,(size_t) blocks * sizeof(float)));
      shmem_reduce_min_kernel<<<blocks, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float) / 2>>>
        (d_intermediate, d_in, size_);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      size_ = blocks;
      blocks = (blocks >> bitMaxThreadsPerBlock) + 1;
      if(blocks == 1)
      {
        shmem_reduce_min_kernel<<<1, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float) / 2>>>
          (d_out_min, d_intermediate, size_);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      }
      else 
      {
        checkCudaErrors(cudaMalloc(&d_intermediate2,(size_t) blocks * sizeof(float)));
        shmem_reduce_min_kernel<<<blocks, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float) / 2>>>
          (d_intermediate2, d_intermediate, size_); 
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        shmem_reduce_min_kernel<<<1, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float) / 2>>>
          (d_out_min, d_intermediate2, blocks); 
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      } 
    }
  
    blocks = (size >> bitMaxThreadsPerBlock) + 1;
    if(blocks == 1)
    {
      shmem_reduce_max_kernel<<<1, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float) / 2>>>
        (d_out_max, d_in, size);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }
    else
    {
      shmem_reduce_max_kernel<<<blocks, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float) / 2>>>
        (d_intermediate, d_in, size);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      size = blocks;
      blocks = (blocks >> bitMaxThreadsPerBlock) + 1;
      if(blocks == 1)
      {
        shmem_reduce_max_kernel<<<1, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float) / 2>>>
          (d_out_max, d_intermediate, size);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaFree(d_intermediate));
      }
      else 
      {
        shmem_reduce_max_kernel<<<blocks, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float) / 2>>>
          (d_intermediate2, d_intermediate, size); 
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        shmem_reduce_max_kernel<<<1, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float) / 2>>>
          (d_out_max, d_intermediate2, blocks);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaFree(d_intermediate));
        checkCudaErrors(cudaFree(d_intermediate2));
      } 
    } 
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{

float * d_out_min, * d_out_max;


checkCudaErrors(cudaMalloc((void **) &d_out_min, sizeof(float)));
checkCudaErrors(cudaMalloc((void **) &d_out_max, sizeof(float)));


min_max_finding(d_out_min, d_out_max, d_logLuminance, (int) numCols * numRows);

checkCudaErrors(cudaMemcpy(&min_logLum, d_out_min, sizeof(float), cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(&max_logLum, d_out_max, sizeof(float), cudaMemcpyDeviceToHost));


float *h_logLuminance = (float *) malloc(sizeof(float)*numRows*numCols); 
checkCudaErrors(cudaMemcpy(h_logLuminance, d_logLuminance, numRows*numCols*sizeof(float), cudaMemcpyDeviceToHost));

unsigned int *h_cdf = (unsigned int *) malloc(sizeof(unsigned int)*numBins);
checkCudaErrors(cudaMemcpy(d_cdf, h_cdf, sizeof(unsigned int) * numBins, cudaMemcpyHostToDevice));

float logLumMin, logLumMax;
logLumMin = 0.f;
logLumMax = 1.f;

int ii = 0;

for (size_t i = 0; i < numCols * numRows; ++i) {
    ii = (h_logLuminance[i]<logLumMin) ? i : ii;
    logLumMin = std::min(h_logLuminance[i], logLumMin);
    logLumMax = std::max(h_logLuminance[i], logLumMax);
}
printf("h[%d] = %f\n",ii,h_logLuminance[ii]);
printf("cpu min:%f gpu min:%f\n",logLumMin,min_logLum);
printf("cpu max:%f gpu max:%f\n",logLumMax,max_logLum);
// referenceCalculation(h_logLuminance, h_cdf, numRows, numCols, numBins, &logLumMin, &logLumMax);


checkCudaErrors(cudaFree(d_out_min));
checkCudaErrors(cudaFree(d_out_max));
free(h_logLuminance);
free(h_cdf);


  
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
