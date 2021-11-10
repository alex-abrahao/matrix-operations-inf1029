#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "matrix_lib.h"


/*
NVIDIA CUDA para operação com matrizes - T4
*/
static int threadsPerBlock = 256;
static int maxBlocksPerGrid = 4096;

int set_grid_size(int threads_per_block, int max_blocks_per_grid) {
    if (threads_per_block > 1024 || max_blocks_per_grid > 65535) {
        threadsPerBlock = 256;
		maxBlocksPerGrid = 4096;
		return 0;
	}

    threadsPerBlock = threads_per_block;
    maxBlocksPerGrid = max_blocks_per_grid;
    return 1;
}


static int test_matrix(struct matrix *matrix) {
    if (matrix == NULL) //testa se a matrix é NULL
        return 0;

    if (matrix->height <= 0 || matrix->width <= 0) //Testa se a matrix tem dimensões
        return 0;

    // Testa se a matriz tem dimensões multiplas de 8
    if ((matrix->height % 8 != 0) || (matrix->width % 8 != 0))
        return 0;

    return 1;
}

__global__
void scalar_thread(int n, float * d_x, float scalar) {
    unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; i < n; i += stride) {
        d_x[i] *= scalar;
    }
}


int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    if (test_matrix(matrix) == 0) //testa a matrix de input
        return 0;
   
    int blockSize = threadsPerBlock, matrixSize = matrix->height * matrix->width;
	int numBlocks = (matrixSize + blockSize - 1) / blockSize;
	if (numBlocks > maxBlocksPerGrid)
        numBlocks = maxBlocksPerGrid;

	scalar_thread<<<numBlocks, blockSize>>>(matrixSize, matrix->d_rows, scalar_value);

    // Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

    return 1;
}



void matrix_thread(void * thread_data) {

    // struct thread_data_matrix_mult * data = (struct thread_data_matrix_mult *) thread_data;
    // Matrix * matrixA = data->matrixA, * matrixB = data->matrixB, * matrixC = data->matrixC;
    // unsigned long i, j, k, indexA, indexB, indexC;
    // unsigned long startLine = data->startLine, lines_chunk = data->lines_chunk;

    // __m256 a, b, c, scalar_a_b;
    // float * addrC;

    // indexA = startLine * matrixA->width;
    // for (i = startLine; i < startLine + lines_chunk; i++) { // para cada linha de A
    //     indexB = 0; // percorre B desde o inicio
    //     for (j = 0; j < matrixA->width; j++) { // para cada elemento A[i][j]

    //         indexC = i * matrixC->width; // percorre C desde o inicio da linha i
    //         a = _mm256_set1_ps(matrixA->rows[indexA]);

    //         for (k = 0; k < matrixC->width; k += 8) { // percorre a linha C[i]
    //             addrC = matrixC->rows + indexC;
    //             c = _mm256_load_ps(addrC);
    //             b = _mm256_load_ps(matrixB->rows + indexB);
                
    //             scalar_a_b = _mm256_fmadd_ps(a, b, c);
    //             _mm256_store_ps(addrC, scalar_a_b);

    //             indexB += 8;
    //             indexC += 8;
    //         }
    //         indexA++; // so passa para o proximo elemento de A quando termina de preencher a linha de C
    //     }
    // }

    // pthread_exit(thread_data);
}



int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC) {
    // unsigned long i, j, k, indexA, indexB, indexC;

    // __m256 a, b, c, scalar_a_b;
    // float * addrC;

    if (test_matrix(matrixA) == 0 || test_matrix(matrixB) == 0) // testa a matrix
        return 0;

    // calcula o produto de uma matriz A (m x n) por uma matriz B (n x p),
    // armazenando o resultado na matriz C (m x p), previamente criada

    // testa se é possivel fazer a multiplicação entre as matrizes (m x n * n x p)
    if (matrixA->width != matrixB->height)
        return 0;

    if (matrixA->height != matrixC->height || matrixB->width != matrixC->width) 
        return 0;

    // for (i = 0; i < matrixC->height * matrixC->width; i += 8) {
    //     // Preenche C com zeros
    //     c = _mm256_set1_ps(0);
    //     _mm256_store_ps(&matrixC->rows[i], c);
    // }

    // unsigned long lines_chunk = matrixA->height / nThreads;
    // void *status;
    // int rc;
    // struct thread_data_matrix_mult thread_data_array[nThreads]; 
    // pthread_t tScalar[nThreads];

    // for (i = 0; i < nThreads; i++) {

    //     thread_data_array[i].thread_id = i;
    //     thread_data_array[i].lines_chunk = lines_chunk;
    //     thread_data_array[i].startLine = i * lines_chunk;
    //     thread_data_array[i].matrixA = matrixA;
    //     thread_data_array[i].matrixB = matrixB;
    //     thread_data_array[i].matrixC = matrixC;
    //     rc = pthread_create(&tScalar[i], NULL, matrix_thread, (void *) &thread_data_array[i]);
        
    //     if (rc) {
    //         printf("ERROR: return code from pthread_create() is %d\n", rc);
    //         return 0;
    //     }
    // }

    // for (i = 0; i < nThreads; i++) {
    //     rc = pthread_join(tScalar[i], &status);

    //     if (rc) {
    //         printf("ERROR: return code from pthread_join() is %d\n", rc);
    //           return 0;
    //     }
    // }

    // Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

    return 1;
}

