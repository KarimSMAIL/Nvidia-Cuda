#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>

#include <cuda.h>
#include <cuda_runtime.h>
#define BLOCK_WIDTH 32
#define TAILLE 4096

#define gettime(t) clock_gettime(CLOCK_MONOTONIC_RAW, t)
#define get_sub_seconde(t) (1e-9*(double)t.tv_nsec)
/** return time in second
*/
double get_elapsedtime(void)
{
  struct timespec st;
  int err = gettime(&st);
  if (err !=0) return 0;
  return (double)st.tv_sec + get_sub_seconde(st);
}

void init(double* A, double* B, double* C, int size)
{
  int i = 0;

  srand(2020);

  for(i = 0; i < size; i++)
  {
    A[i] = i * 1.;
    B[i] = i * 1.;
    C[i] = 0.0;
  }
}

void add(double* A, double* B, double* C, int size)
{
  int i = 0;

  for(i = 0; i < size; i++)
  {
    C[i] = A[i] + B[i];
  }
}

// QUESTION 4
__global__
void AddVecKernel(double* A, double* B, double* C, int N)
{
  // QUESTION 6
  int col    = threadIdx.x + blockDim.x * blockIdx.x;
  // FIN QUESTION 6

  // QUESTION 7
  if((col < N))
  {
    C[col] = A[col] + B[col];
  }
  // FIN QUESTION 7
}
// FIN QUESTION 4

int main(int argc, char** argv){
  int N;

  double *A;
  double *B;
  double *C;
  double *C_bis;

  double t0 = 0., t1 = 0., duration = 0.;

  N = (argc < 2)?1000:atoi(argv[1]);
  fprintf(stdout, "Vectors addition\n  Size: %d\n", N);

  // Memory allocation
  A = (double*) malloc(sizeof(double) * N);
  B = (double*) malloc(sizeof(double) * N);
  C = (double*) malloc(sizeof(double) * N);
  C_bis = (double*) malloc(sizeof(double) * N);

  // Value initialization
  init(A, B, C, N);

  // QUESTION 8
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  //FIN QUESTION 8

  // QUESTION 1
  double *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeof(double) * N);
  cudaMalloc(&d_B, sizeof(double) * N);
  cudaMalloc(&d_C, sizeof(double) * N);
  // FIN QUESTION 1

  // QUESTION 2
  cudaMemcpy(d_A, A, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, sizeof(double) * N, cudaMemcpyHostToDevice);
  // FIN QUESTION 2

  // QUESTION 3
  int nbBlocks = N / BLOCK_WIDTH;
  if(N % BLOCK_WIDTH) nbBlocks++;
  dim3 gridSize(nbBlocks);
  dim3 blockSize(BLOCK_WIDTH);
  // FIN QUESTION 3

  // QUESTION 4
  cudaEventRecord(start); // QUESTION 8
  AddVecKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
  cudaEventRecord(stop); // QUESTION 8
  // FIN QUESTION 4

  // QUESTION 5
  cudaMemcpy(C, d_C, sizeof(double) * N, cudaMemcpyDeviceToHost);
  // FIN QUESTION 5

  // QUESTION 8
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Vecteur %d\n\tTemps: %f s\n", N, milliseconds/1000);
  // FIN QUESTION 8

  // Compute multiplication
  t0 = get_elapsedtime();
  add(A, B, C_bis, N);
  t1 = get_elapsedtime();

  for(int i = 0; i < N; ++i)
  {
    if(C[i] != C_bis[i])
    {
      fprintf(stderr, "FATAL ERROR ! [%d : GPU: %f != CPU: %f]\n", i, C[i], C_bis[i]);
      exit(-1);
    }
  }

  // Pretty print
  duration = (t1 - t0);
  uint64_t nb_op = N;
  fprintf(stdout, "Performance results: \n");
  fprintf(stdout, "  Time: %lf s\n", duration);
  fprintf(stdout, "  MFlops: %.2f\n", (nb_op / duration)*1E-6);

  free(A);
  free(B);
  free(C);

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return 0;
}
