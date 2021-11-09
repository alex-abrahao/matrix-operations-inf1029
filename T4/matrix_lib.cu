#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
#include "matrix_lib.h"
#include <pthread.h>

struct thread_data_scalar {
    int thread_id;
    unsigned long lines_chunk;
    unsigned long columns;
    float * chunkStart;
    __m256 scalar_value;
};

struct thread_data_matrix_mult {
    int thread_id;
    unsigned long lines_chunk;
    unsigned long startLine;
    Matrix * matrixA;
    Matrix * matrixB;
    Matrix * matrixC;
};



/*
Módulo avançado (AVX/FMA) com Threads para operação com matrizes - T3
*/
static int nThreads = 1;

void set_number_threads(int num_threads) {
    nThreads = num_threads;
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


void * scalar_thread(void * thread_data) {
    unsigned long i;
    
    struct thread_data_scalar * data = (struct thread_data_scalar *) thread_data;
    unsigned long lines_chunk = data->lines_chunk, columns = data->columns;
    float * chunkStart = data->chunkStart;

    __m256 vet1 = data->scalar_value, vet2, res;

    for (i = 0; i < lines_chunk * columns; i += 8) {
        vet2 = _mm256_load_ps(&chunkStart[i]);
        res	= _mm256_mul_ps(vet2, vet1);
        _mm256_store_ps(&chunkStart[i], res);
    }
    
    pthread_exit(thread_data);
}


int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
     

    if (test_matrix(matrix) == 0) //testa a matrix de input
        return 0;

    if (matrix->height % nThreads != 0) {
        printf("ERROR: Threads are not a multiple of the number of lines\n");
        return 0;
    }
   
    void *status;
    unsigned long i = 0, lines_chunk = matrix->height / nThreads, chunk_size = lines_chunk * matrix->width;
    int rc;
    struct thread_data_scalar thread_data_array[nThreads]; 
    pthread_t tScalar[nThreads];

    __m256 vet1 = _mm256_set1_ps(scalar_value);

    for (i = 0; i < nThreads; i++) {

        thread_data_array[i].thread_id = i;
        thread_data_array[i].lines_chunk = lines_chunk;
        thread_data_array[i].columns = matrix->width;
        thread_data_array[i].scalar_value = vet1;
        thread_data_array[i].chunkStart = (matrix->rows + i * chunk_size);
        rc = pthread_create(&tScalar[i], NULL, scalar_thread, (void *) &thread_data_array[i]);
        
        if (rc) {
            printf("ERROR: return code from pthread_create() is %d\n", rc);
            return 0;
        }
    }

    for (i = 0; i < nThreads; i++) {
        rc = pthread_join(tScalar[i], &status);

        if (rc) {
            printf("ERROR: return code from pthread_join() is %d\n", rc);
            return 0;
        }
    }

    return 1;
}



void * matrix_thread(void * thread_data) {

    struct thread_data_matrix_mult * data = (struct thread_data_matrix_mult *) thread_data;
    Matrix * matrixA = data->matrixA, * matrixB = data->matrixB, * matrixC = data->matrixC;
    unsigned long i, j, k, indexA, indexB, indexC;
    unsigned long startLine = data->startLine, lines_chunk = data->lines_chunk;

    __m256 a, b, c, scalar_a_b;
    float * addrC;

    indexA = startLine * matrixA->width;
    for (i = startLine; i < startLine + lines_chunk; i++) { // para cada linha de A
        indexB = 0; // percorre B desde o inicio
        for (j = 0; j < matrixA->width; j++) { // para cada elemento A[i][j]

            indexC = i * matrixC->width; // percorre C desde o inicio da linha i
            a = _mm256_set1_ps(matrixA->rows[indexA]);

            for (k = 0; k < matrixC->width; k += 8) { // percorre a linha C[i]
                addrC = matrixC->rows + indexC;
                c = _mm256_load_ps(addrC);
                b = _mm256_load_ps(matrixB->rows + indexB);
                
                scalar_a_b = _mm256_fmadd_ps(a, b, c);
                _mm256_store_ps(addrC, scalar_a_b);

                indexB += 8;
                indexC += 8;
            }
            indexA++; // so passa para o proximo elemento de A quando termina de preencher a linha de C
        }
    }

    pthread_exit(thread_data);
}



int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC) {
    unsigned long i, j, k, indexA, indexB, indexC;

    __m256 a, b, c, scalar_a_b;
    float * addrC;

    if (test_matrix(matrixA) == 0 || test_matrix(matrixB) == 0) // testa a matrix
        return 0;

    if (matrixA->height % nThreads != 0) {
        printf("ERROR: Threads are not a multiple of the number of lines\n");
        return 0;
    }
    // calcula o produto de uma matriz A (m x n) por uma matriz B (n x p),
    // armazenando o resultado na matriz C (m x p), previamente criada

    // testa se é possivel fazer a multiplicação entre as matrizes (m x n * n x p)
    if (matrixA->width != matrixB->height)
        return 0;

    if (matrixA->height != matrixC->height || matrixB->width != matrixC->width) 
        return 0;

    for (i = 0; i < matrixC->height * matrixC->width; i += 8) {
        // Preenche C com zeros
        c = _mm256_set1_ps(0);
        _mm256_store_ps(&matrixC->rows[i], c);
    }

    unsigned long lines_chunk = matrixA->height / nThreads;
    void *status;
    int rc;
    struct thread_data_matrix_mult thread_data_array[nThreads]; 
    pthread_t tScalar[nThreads];

    for (i = 0; i < nThreads; i++) {

        thread_data_array[i].thread_id = i;
        thread_data_array[i].lines_chunk = lines_chunk;
        thread_data_array[i].startLine = i * lines_chunk;
        thread_data_array[i].matrixA = matrixA;
        thread_data_array[i].matrixB = matrixB;
        thread_data_array[i].matrixC = matrixC;
        rc = pthread_create(&tScalar[i], NULL, matrix_thread, (void *) &thread_data_array[i]);
        
        if (rc) {
            printf("ERROR: return code from pthread_create() is %d\n", rc);
            return 0;
        }
    }

    for (i = 0; i < nThreads; i++) {
        rc = pthread_join(tScalar[i], &status);

        if (rc) {
            printf("ERROR: return code from pthread_join() is %d\n", rc);
              return 0;
        }
    }

    return 1;
}

