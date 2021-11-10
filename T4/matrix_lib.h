typedef struct matrix {
    unsigned long int height;
    unsigned long int width;
    float *h_rows;
    float *d_rows;
} Matrix;


// Multiplica a matriz por um escalar
int scalar_matrix_mult(float scalar_value, struct matrix *matrix); 

// Faz a multiplicação entre duas matrizes
int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC);

int set_grid_size(int threads_per_block, int max_blocks_per_grid);