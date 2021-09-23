#include <stdlib.h>
#include <immintrin.h>
#include "matrix_lib.h"

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

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
     
     if (test_matrix(matrix) == 0) //testa a matrix de input
        return 0;
   
    unsigned long m = matrix->height, n = matrix->width;
    unsigned long i = 0, j = 0;
    __m256 v1 = _mm256_set1_ps(scalar_value);
	__m256 v2;
	__m256 res;

    for (i = 0; i < m * n; i += 8) {
		v2 = _mm256_load_ps(&matrix->rows[i]);
		res	= _mm256_mul_ps(v2, v1);
		_mm256_store_ps(&matrix->rows[i], res);			
	}

	return 1;
}




int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC) {
    unsigned long i, j, k, indexA, indexB, indexC;

    if (test_matrix(matrixA) == 0 || test_matrix(matrixB) == 0) // testa a matrix
        return 0;

    // calcula o produto de uma matriz A (m x n) por uma matriz B (n x p),
    // armazenando o resultado na matriz C (m x p), previamente criada

    // testa se é possivel fazer a multiplicação entre as matrizes (m x n * n x p)
    if (matrixA->width != matrixB->height)
        return 0;

    if (matrixA->height != matrixC->height || matrixB->width != matrixC->width) 
        return 0;

    for (i = 0; i < matrixC->height * matrixC->width; i++)
        matrixC->rows[i] = 0.0; // Preenche C com zeros

    for (i = 0, indexA = 0; i < matrixA->height; i++) { // para cada linha de A
        indexB = 0; // percorre B desde o inicio
        for (j = 0; j < matrixA->width; j++) { // para cada elemento A[i][j]
            indexC = i * matrixC->width; // percorre C desde o inicio da linha i
            for (k = 0; k < matrixC->width; k++) { // percorre a linha C[i]
                matrixC->rows[indexC] += matrixA->rows[indexA] * matrixB->rows[indexB];

                indexB++;
                indexC++;
            }
            indexA++; // so passa para o proximo elemento de A quando termina de preencher a linha de C
        }
    }

    return 1;
}

// programa base abaixo

// int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC) {
//     unsigned long i, j, k, indexA, indexB, indexC;
//     float sum;

//     if (test_matrix(matrixA) == 0 || test_matrix(matrixB) == 0) // testa a matrix
//         return 0;

//     // calcula o produto de uma matriz A (m x n) por uma matriz B (n x q),
//     // armazenando o resultado na matriz C (m x q), previamente criada


//     // testa se é possivel fazer a multiplicação entre as matrizes (2x1 * 1x5)
//     if (matrixA->width != matrixB->height) 
//         return 0;

//     if (matrixA->height != matrixC->height || matrixB->width != matrixC->width) 
//         return 0;

//     for (i = 0; i < matrixA->height; i++) {
//         for (k = 0; k < matrixB->width; k++) {
//             sum = 0.0;
//             indexC = i * matrixA->width + k;
//             for (j = 0; j < matrixA->width; j++) {
//                 indexA = i * matrixA->width + j;
//                 indexB = j * matrixB->width + k;
//                 sum += matrixA->rows[indexA] * matrixB->rows[indexB];
//             }
//             matrixC->rows[indexC] = sum;
//         }
//     }

//     return 1;
// }
