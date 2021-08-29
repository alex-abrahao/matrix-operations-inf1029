#include<stdlib.h>
#include"matrix_lib.h"

static int test_matrix(struct matrix *matrix) {
    if (matrix == NULL) //testa se a matrix é NULL
        return 0;

    if (matrix->height <= 0 || matrix->width <= 0) //Testa se a matrix tem dimensões
        return 0;

    return 1;
}

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    unsigned long i;
    if (test_matrix(matrix) == 0) //testa a matrix de input
        return 0;
    
    for (i = 0; i < matrix->height * matrix->width; i++) {
        matrix->rows[i] *= scalar_value;
    }
    return 1;
}

int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC) {
    unsigned long i, j, k, indexA, indexB, indexC;
    float sum;

    if (test_matrix(matrixA) == 0 || test_matrix(matrixB) == 0) // testa a matrix
        return 0;

    // calcula o produto de uma matriz A (m x n) por uma matriz B (n x q),
    // armazenando o resultado na matriz C (m x q), previamente criada


    // testa se é possivel fazer a multiplicação entre as matrizes (2x1 * 1x5)
    if (matrixA->width != matrixB->height) 
        return 0;

    if (matrixA->height != matrixC->height || matrixB->width != matrixC->width) 
        return 0;

    for (i = 0; i < matrixA->height; i++) {
        for (k = 0; k < matrixB->width; k++) {
            sum = 0.0;
            indexC = i * matrixA->width + k;
            for (j = 0; j < matrixA->width; j++) {
                indexA = i * matrixA->width + j;
                indexB = j * matrixB->width + k;
                sum += matrixA->rows[indexA] * matrixB->rows[indexB];
            }
            matrixC->rows[indexC] = sum;
        }
    }

    return 1;
}
