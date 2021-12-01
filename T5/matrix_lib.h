typedef struct matrix {
    unsigned long int height;
    unsigned long int width;
    float *vh_rows;
    void *ve_rows;
} Matrix;


// Multiplica a matriz por um escalar
int scalar_matrix_mult(float scalar_value, struct matrix *matrix); 

// Faz a multiplicação entre duas matrizes
int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC);

void set_ve_execution_node(int num_node);

void set_number_threads(int num_threads);

int init_proc_ve_node();

int close_proc_ve_node();

int load_ve_matrix(struct matrix *matrix);

int unload_ve_matrix(struct matrix *matrix);

int sync_vh_ve_matrix(struct matrix *matrix);

int sync_ve_vh_matrix(struct matrix *matrix);
