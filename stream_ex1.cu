#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys.time.h>

#define NSTREAMS 2

__host__ void init(float *out, int sz)
{
    for(int i = 0 ; i < sz ; i++)
    {
	out[i] = sin(float(i));
    }
}

__host__ void verif(float *out, int sz)
{
    float err = 0.;

    for(int i = 0 ; i < sz ; i++)
    {
	err += abs(out[i] - exp( - abs(sin(float(i))) ));
    }

    if (err/sz < 1.e-4)
    {
	printf("TEST PASSED (error %3.f < 1.e-4)\n", err/sz);
    }
    else
    {
	printf("TEST FAILED (error %3.f > 1.e-4)\n", err/sz);
    }
}

__global__ void MyKernel(float *out, float *in, int sz)
{
	    int i = blockIdx.x * blockDim.x + threadIdx.x ;
      if(i < sz)
      {
	      out[i] = exp( - abs(in[i]) );
      }
}

int main(int argc, char **argv)
{
    int size, nblocks, nthreads = 128;

    if (argc == 2)
	size = atoi(argv[1]);
    else
	size = 1000;

    nblocks = (size + nthreads-1) / nthreads;

    printf("size                : %d\n", size);
    printf("NSTREAMS            : %d\n", NSTREAMS);
    printf("Taille des tableaux : %d\n", NSTREAMS*size);
    printf("nblocks             : %d\n", nblocks);
    printf("nthreads            : %d\n", nthreads);

    float *hostPtr, *inputDevPtr, *outputDevPtr;

    struct timeval time1, time2;

    gettimeofday(&time1, NULL);

    cudaMalloc((void**)&inputDevPtr, NSTREAMS * size * sizeof(float));
    cudaMalloc((void**)&outputDevPtr, NSTREAMS * size * sizeof(float));

    cudaMallocHost((void**)&hostPtr, NSTREAMS * size * sizeof(float));

    init(hostPtr, NSTREAMS * size);

    cudaStream_t stream[NSTREAMS];

    for (int i = 0; i < NSTREAMS; ++i)
    	cudaStreamCreate(&stream[i]);

    for (int i = 0; i < NSTREAMS; ++i)
    	cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size, size*sizeof(float), cudaMemcpyHostToDevice, stream[i]);

    for (int i = 0; i < NSTREAMS; ++i)
    	MyKernel<<<nblocks, nthreads, 0, stream[i]>>> (outputDevPtr + i * size, inputDevPtr + i * size, size);

    for (int i = 0; i < NSTREAMS; ++i)
    	cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size, size*sizeof(float), cudaMemcpyDeviceToHost, stream[i]);

    cudaThreadSynchronize();

    for (int i = 0; i < NSTREAMS; ++i)
    	cudaStreamDestroy(stream[i]);

    verif(hostPtr, NSTREAMS*size);

    cudaFree(inputDevPtr);
    cudaFree(outputDevPtr);

    cudaFreeHost(hostPtr);

    gettimeofday(time2, NULL);

    printf("KERNEL EXECUTION TIME:-- %d s %d us --\n", time2.tv_sec - time1.tv_sec, time2.tv_usec - time1.tv_usec);


    return 0;
}
