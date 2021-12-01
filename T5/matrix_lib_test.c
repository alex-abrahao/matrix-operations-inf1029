#include "matrix_lib.h"
#include "timer.h"
#include <stdio.h>
#include <stdlib.h>

 
/*
typedef struct matrix {
    unsigned long int height;
    unsigned long int width;
    float *vh_rows;
    void *ve_rows;
} Matrix;
*/

// Inicializa as matrizes em que irão ser realizadas operações
static struct matrix * init_matrix(unsigned long height, unsigned long width) {
    // Aloca espaço para matriz e da o tamanho para ela
    struct matrix * matrix = (Matrix *) malloc(sizeof(Matrix));
    if (matrix == NULL) {
        printf("Problema ao alocar matriz\n");
        exit(1);
    }
    matrix->height = height;
    matrix->width = width;
    matrix->vh_rows = (float *) malloc(matrix->height * matrix->width * (sizeof(float)));

    if (matrix->vh_rows == NULL) {
        printf("Problema ao alocar linhas da matriz no host\n");
        exit(1);
    }

    return matrix;
}

static void fillMatrix(struct matrix *matrix, const char * fileName) {
    FILE *file;
    
    file = fopen(fileName, "rb");

    if (file == NULL) {
        printf("Erro na abertura do arquivo.\n");
        system("pause");
        exit(1);
    }
    fread(matrix->vh_rows, matrix->height * matrix->width * sizeof(float), 1, file);
    fclose(file);

    if (load_ve_matrix(matrix) == 0) {
        printf("load_ve_matrix (matrix.vh_rows -> matrix.ve_rows) returned error\n");
        exit(1);
    }
}

// Inicializa a matriz resultado de !!!! A * B !!!!
static struct matrix * init_matrixResult(struct matrix *matrixA, struct matrix * matrixB) { 
    // Testa se as matrizes são compativeis
    if (matrixA->width != matrixB->height)  {
        printf("matrixC: widthA != heightB\n");
        exit(1);
    }

    // Aloca espaço para matriz C e da o tamanho para ela
    struct matrix * matrixC = init_matrix(matrixA->height, matrixB->width);

    // Confere o se tudo ocorreu certo na alocação da matriz C em relação ao tamanho
    if (matrixA->height != matrixC->height || matrixB->width != matrixC->width) {
        printf("Erro alocando C\n");
        exit(1);
    }

    return matrixC;
}

static void printMatrix(struct matrix * matrix) {
    int i, j, index;
    for (i = 0; i < matrix->height && i < 255; i++) {
        index = i * matrix->width;
        for (j = 0; j < matrix->width && j < 255; j++)
            printf("%f ", matrix->vh_rows[index++]);
        printf("\n");
    }
    printf("\n\n");
}

static void saveMatrix(struct matrix * matrix, const char * fileName) {
    FILE *file;

    file = fopen(fileName, "wb");

    if (file == NULL) {
        printf("Erro na abertura do arquivo %s.\n", fileName);
        system("pause");
        exit(1);
    }
    // printMatrix(matrix);

    fwrite(matrix->vh_rows, matrix->height * matrix->width * sizeof(float), 1, file);
    fclose(file);
}

// Libera espaço alocado para uma matriz
static void freeMatrix(struct matrix * matrix) {
    if (matrix->ve_rows != NULL) {
        unload_ve_matrix(matrix);
    }
    free(matrix->vh_rows);
    free(matrix);
}

int main(int argc, char* argv[]) {
    struct timeval start, stop, overall_t1, overall_t2;


    // Mark overall start time
    gettimeofday(&overall_t1, NULL);

    if (argc != 12) {
        printf("Parametros incorretos! Exemplo de comando:\n");
        printf("./%s 5.0 8 16 16 8 1 8 floats_256_2.0f.dat floats_256_5.0f.dat result1.dat result2.dat\n", argv[0]);
        return 1;
    }

    /*
    Exemplo de comando:

    ./matrix_lib_test 5.0 8 16 16 8 256 4096 floats_256_2.0f.dat floats_256_5.0f.dat result1.dat result2.dat

    5.0 é o valor escalar que multiplicará a primeira matriz;
    8 é o número de linhas da primeira matriz;
    16 é o número de colunas da primeira matriz;
    16 é o número de linhas da segunda matriz;
    8 é o número de colunas da segunda matriz;

    1 é o número de identificação da VE de execução;
    8 é o número de threads a serem disparadas na VE de execução;

    floats_256_2.0f.dat é o nome do arquivo de floats que será usado para carregar a primeira matriz;
    floats_256_5.0f.dat é o nome do arquivo de floats que será usado para carregar a segunda matriz;

    result1.dat é o nome do arquivo de floats onde o primeiro resultado será armazenado;
    result2.dat é o nome do arquivo de floats onde o segundo resultado será armazenado.
    */

    float scalar;
    unsigned long heightA, widthA, heightB, widthB;
    const char * floatsA, * floatsB, * floatsResult1, * floatsResult2;
    int ve_node, numThreads;
    
    scalar = strtof(argv[1], NULL); // Conversão p/ float

    // Conversões p/ unsigned long, em base 10
    heightA = strtoul(argv[2], NULL, 10);
    widthA = strtoul(argv[3], NULL, 10);
    heightB = strtoul(argv[4], NULL, 10);
    widthB = strtoul(argv[5], NULL, 10);

    ve_node = atoi(argv[6]);
    numThreads = atoi(argv[7]);

    floatsA = argv[8];
    floatsB = argv[9];

    floatsResult1 = argv[10];
    floatsResult2 = argv[11];

    // Set params and init process
    set_ve_execution_node(ve_node);
    set_number_threads(numThreads);
    init_proc_ve_node();

    // Inicialização das matrizes
    struct matrix * matrixA = init_matrix(heightA, widthA);
    struct matrix * matrixB = init_matrix(heightB, widthB);
    struct matrix * matrixC = init_matrixResult(matrixA, matrixB);

    fillMatrix(matrixA, floatsA);
    fillMatrix(matrixB, floatsB);
    
    /*
    A função scalar_matrix_mult deve ser chamada com os seguintes argumentos: o
    valor escalar fornecido e a primeira matriz (A). O resultado do produto
    (retornado na matriz A) deve ser armazenado em um arquivo binário usando o
    nome do terceiro arquivo de floats.
    */

    int operationResult;
    
    // Mark init start time
    gettimeofday(&start, NULL);

    operationResult = scalar_matrix_mult(scalar, matrixA);

    // Mark init stop time
    gettimeofday(&stop, NULL);
    
    if (operationResult == 0) {
        printf("Multiplicacao escalar com problema\n");
    } else {
        printf("Scalar mult time: %f ms\n", timedifference_msec(start, stop));
        if (sync_ve_vh_matrix(matrixA) == 0) {
            printf("Erro ao copiar a matriz A antes de salvar.\n");
        } else {
            saveMatrix(matrixA, floatsResult1);
        }
    }

    /*
    Depois, a função matrix_matrix_mult deve ser chamada com os seguintes
    argumentos: a matriz A resultante da função scalar_matrix_mult, a segunda
    matriz (B) e a terceira matriz (C). O resultado do produto (retornado na
    matriz C) deve ser armazenado em um arquivo binário com o nome do quarto
    arquivo de floats.
    */

    gettimeofday(&start, NULL);
    operationResult = matrix_matrix_mult(matrixA, matrixB, matrixC);
    gettimeofday(&stop, NULL);

    if (operationResult == 0) {
        printf("Multiplicacao matricial com problema\n");
    } else {
        printf("Matrix mult time: %f ms\n", timedifference_msec(start, stop));
        if (unload_ve_matrix(matrixC) == 0) {
            printf("Erro ao copiar a matriz C antes de salvar.\n");
        } else {
            saveMatrix(matrixC, floatsResult2);
        }
    }

    // Libera matrizes
    freeMatrix(matrixA);
    freeMatrix(matrixB);
    freeMatrix(matrixC);

    // Close VE process
    close_proc_ve_node();

    // Mark overall stop time
    gettimeofday(&overall_t2, NULL);
    // Show elapsed overall time
    printf("Overall time: %f ms\n", timedifference_msec(overall_t1, overall_t2));

    return 0;
}