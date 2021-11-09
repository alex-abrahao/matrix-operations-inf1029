typedef struct matrix {
    unsigned long int height;
    unsigned long int width;
    float *rows;
    
} Matrix;

//typedef struct matrix Matrix; é a mesma coisa so que mais clean ^

// Multiplica a matriz por um escalar
int scalar_matrix_mult(float scalar_value, struct matrix *matrix); 


// Faz a multiplicação entre duas matrizes
int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC);