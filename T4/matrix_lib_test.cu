#include "matrix_lib.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
extern "C" {
#include "timer.h"
}
 
/*
struct matrix {
    unsigned long int height;
    unsigned long int width;
    float *rows;
};

typedef struct matrix Matrix;
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
    matrix->h_rows = (float *) aligned_alloc(32, matrix->height * matrix->width * (sizeof(float)));

    if (matrix->h_rows == NULL) {
        printf("Problema ao alocar linhas da matriz no host\n");
        exit(1);
    }

    cudaError_t cudaError = cudaMalloc(&matrix->d_rows, matrix->height * matrix->width * (sizeof(float)));

    if (cudaError != cudaSuccess) {
        printf("cudaMalloc returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
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
    fread(matrix->h_rows, matrix->height * matrix->width * sizeof(float), 1, file);
    fclose(file);

    cudaError_t cudaError = cudaMemcpy(matrix->d_rows, matrix->h_rows, matrix->height * matrix->width * sizeof(float), cudaMemcpyHostToDevice);

    if (cudaError != cudaSuccess) {
        printf("cudaMemcpy (matrix.h_rows -> matrix.d_rows) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
        exit(1);
    }
}

// Inicializa a matriz resultado de !!!! A * B !!!! preenchida com 0
static struct matrix * init_matrixResult(struct matrix *matrixA, struct matrix * matrixB) { 
    // Testa se as matrizes são compativeis
    if (matrixA->width != matrixB->height)  {
        printf("widthA != heightB\n");
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
    for (i = 0; i < matrix->height; i++) {
        index = i * matrix->width;
        for (j = 0; j < matrix->width; j++)
            printf("%f ", matrix->h_rows[index++]);
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

    fwrite(matrix->h_rows, matrix->height * matrix->width * sizeof(float), 1, file);
    fclose(file);
}

// Libera espaço alocado para uma matriz
static void freeMatrix(struct matrix * matrix) {
    cudaFree(matrix->d_rows);
    free(matrix->h_rows);
    free(matrix);
}

int main(int argc, char* argv[]) {
    struct timeval start, stop, overall_t1, overall_t2;


    // Mark overall start time
    gettimeofday(&overall_t1, NULL);

    if (argc != 12) {
        printf("Parametros incorretos! Exemplo de comando:\n");
        printf("./matrix_lib_test 5.0 8 16 16 8 256 4096 floats_256_2.0f.dat floats_256_5.0f.dat result1.dat result2.dat\n");
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
    256 é o número de threads por bloco a serem disparadas;
    4096 é o número máximo de blocos por GRID a serem usados;

    floats_256_2.0f.dat é o nome do arquivo de floats que será usado para carregar a primeira matriz;
    floats_256_5.0f.dat é o nome do arquivo de floats que será usado para carregar a segunda matriz;

    result1.dat é o nome do arquivo de floats onde o primeiro resultado será armazenado;
    result2.dat é o nome do arquivo de floats onde o segundo resultado será armazenado.
    */

    float scalar;
    unsigned long heightA, widthA, heightB, widthB;
    const char * floatsA, * floatsB, * floatsResult1, * floatsResult2;
    int numThreads, maxBlocks;
    cudaError_t cudaError;
    
    scalar = strtof(argv[1], NULL); // Conversão p/ float

    // Conversões p/ unsigned long, em base 10
    heightA = strtoul(argv[2], NULL, 10);
    widthA = strtoul(argv[3], NULL, 10);
    heightB = strtoul(argv[4], NULL, 10);
    widthB = strtoul(argv[5], NULL, 10);

    numThreads = atoi(argv[6]);
    maxBlocks = atoi(argv[7]);

    floatsA = argv[8];
    floatsB = argv[9];

    floatsResult1 = argv[10];
    floatsResult2 = argv[11];

    // GRID Size
    if (set_grid_size(numThreads, maxBlocks) != 0) {
        printf("Numero de threads por bloco: %d\nMax de blocos por GRID: %d\n", numThreads, maxBlocks);
    } else {
        printf("Erro: Parametros de threads ou max de blocos incorreto. Utilizando valores padroes.\n");
    }

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

    cudaError = cudaMemcpy(matrixA->h_rows, matrixA->d_rows, matrixA->height * matrixA->width * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (operationResult == 0) {
        printf("Multiplicacao escalar com problema\n");
    } else if (cudaError != cudaSuccess) {
        printf("cudaMemcpy (matrix.d_rows -> matrix.h_rows) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
        exit(1);
    } else {
        printf("Scalar mult time: %f ms\n", timedifference_msec(start, stop));
        saveMatrix(matrixA, floatsResult1);
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

    cudaError = cudaMemcpy(matrixC->h_rows, matrixC->d_rows, matrixC->height * matrixC->width * sizeof(float), cudaMemcpyDeviceToHost);

    if (operationResult == 0) {
        printf("Multiplicacao matricial com problema\n");
    } else if (cudaError != cudaSuccess) {
        printf("cudaMemcpy (matrix.d_rows -> matrix.h_rows) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaError), cudaError, __LINE__);
        exit(1);
    } else {
        printf("Matrix mult time: %f ms\n", timedifference_msec(start, stop));
        saveMatrix(matrixC, floatsResult2);
    }

    // Libera matrizes
    freeMatrix(matrixA);
    freeMatrix(matrixB);
    freeMatrix(matrixC);

    // Mark overall stop time
    gettimeofday(&overall_t2, NULL);
    // Show elapsed overall time
    printf("Overall time: %f ms\n", timedifference_msec(overall_t1, overall_t2));

    return 0;
}