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
   
    unsigned long i = 0, j = 0;
    __m256 vet1 = _mm256_set1_ps(scalar_value), vet2, res;

    for (i = 0; i < matrix->height * matrix->width; i += 8) {
		vet2 = _mm256_load_ps(&matrix->rows[i]);
		res	= _mm256_mul_ps(vet2, vet1);
		_mm256_store_ps(&matrix->rows[i], res);			
	}

	return 1;
}

int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC) {
    unsigned long i, j, k, indexA, indexB, indexC;

    __m256 a, b, c, scalar_a_b;
    float * addrC;

    if (test_matrix(matrixA) == 0 || test_matrix(matrixB) == 0) // testa a matrix
        return 0;

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

    for (i = 0, indexA = 0; i < matrixA->height; i++) { // para cada linha de A
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

    return 1;
}
