#include <stdlib.h>
#include <stdio.h>
#include <ve_offload.h>
#include "matrix_lib.h"


/*
NEC Vector Engine - T5
*/
static int ve_execution_node = 0;
static int nThreads = 1;
static struct veo_proc_handle * process = NULL;
static uint64_t library_handle = 0;

void set_ve_execution_node(int num_node) {
    if (num_node < 0 || num_node > 3) {
        ve_execution_node = 0;
        return;
    }

    ve_execution_node = num_node;
}

void set_number_threads(int num_threads) {
    if (num_threads < 0) {
        nThreads = 1;
        return;
    }

    nThreads = num_threads <= 8 ? num_threads : 8;
}

int init_proc_ve_node() {
    process = veo_proc_create(ve_execution_node);
    if (process == NULL) return 0;

    library_handle = veo_load_library(process, "./matrix_lib_ve.so");

    if (library_handle == 0) {
        veo_proc_destroy(process);
        process = NULL;
        return 0;
    }
    return 1;
}

int close_proc_ve_node() {
    int rc = veo_unload_library(process, library_handle);
    if (rc != 0) {
        return 0;
    }
    library_handle = 0;
    rc = veo_proc_destroy(process);
    if (rc == -1) {
        return 0;
    }
    process = NULL;
    return 1;
}

int load_ve_matrix(struct matrix *matrix) {
    return 0;
}

int unload_ve_matrix(struct matrix *matrix) {
    return 0;
}

int sync_vh_ve_matrix(struct matrix *matrix) {
    return 0;
}

int sync_ve_vh_matrix(struct matrix *matrix) {
    return 0;
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

void scalar_thread(unsigned long n, float * d_x, float scalar) {
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

    // scalar_thread<<<numBlocks, blockSize>>>(matrixSize, matrix->d_rows, scalar_value);

    // Wait for GPU to finish before accessing on host
    // cudaDeviceSynchronize();

    return 1;
}


void matrix_thread(unsigned long n, float * d_A, float * d_B, float * d_C, unsigned long widthA, unsigned long widthB, unsigned long widthC) {
    unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    unsigned long j, k, lineA, lineC, colC, endA, indexB;

    for (; i < n; i += stride) {
        lineC = i / widthC;
        colC = i % widthC;
        lineA = lineC * widthA;
        endA = lineA + widthA;
        
        d_C[i] = 0.0;
        
        for (j = lineA, k = 0; j < endA; j++, k++) {
            indexB = k * widthB + colC;
            d_C[i] += d_A[j] * d_B[indexB];
        }
    }
}

int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC) {
    if (test_matrix(matrixA) == 0 || test_matrix(matrixB) == 0) // testa a matrix
        return 0;

    // calcula o produto de uma matriz A (m x n) por uma matriz B (n x p),
    // armazenando o resultado na matriz C (m x p), previamente criada

    // testa se é possivel fazer a multiplicação entre as matrizes (m x n * n x p)
    if (matrixA->width != matrixB->height)
        return 0;

    if (matrixA->height != matrixC->height || matrixB->width != matrixC->width) 
        return 0;

    int blockSize = threadsPerBlock, matrixSize = matrixC->height * matrixC->width;
    int numBlocks = (matrixSize + blockSize - 1) / blockSize;
    if (numBlocks > maxBlocksPerGrid)
        numBlocks = maxBlocksPerGrid;

    // matrix_thread<<<numBlocks, blockSize>>>(matrixSize, matrixA->d_rows, matrixB->d_rows, matrixC->d_rows, matrixA->width, matrixB->width, matrixC->width);

    // Wait for GPU to finish before accessing on host
    // cudaDeviceSynchronize();

    return 1;
}

