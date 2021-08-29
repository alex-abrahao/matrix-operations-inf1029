#include "matrix_lib.h"
#include "timer.h"

#include <stdio.h>
#include <stdlib.h>
 
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
    struct matrix * matrix = malloc(sizeof(Matrix));
    matrix->height = height;
    matrix->width = width;
    matrix->rows = (float *) malloc(matrix->height * matrix->width * (sizeof(float)));

    return matrix;
}

static void fillMatrix(struct matrix *matrix, const char * fileName) {
    FILE *f;

    // Tenho que pensar um pouco vou descansar vemos isso amanhã

    // f = fopen("comets$system:floats_256_2.0f.dat", "rb");
    f = fopen(fileName, "rb");
    if (!f) {
        printf("Erro na abertura do arquivo.\n");
        exit(1);
    }

    // TODO: preencher matriz com arquivo f

}

// Inicializa a matriz resultado de !!!! A * B !!!! preenchida com 0
static struct matrix * init_matrixResult(struct matrix *matrixA, struct matrix * matrixB) { 
    int i;
    
    // Testa se as matrizes são compativeis
    if (matrixA->width != matrixB->height) 
        exit(1);

    // Aloca espaço para matriz C e da o tamanho para ela
    struct matrix * matrixC = init_matrix(matrixB->height, matrixA->width);

    // Confere o se tudo ocorreu certo na alocação da matriz C em relação ao tamanho
    if (matrixA->height != matrixC->height || matrixB->width != matrixC->width) 
        exit(1);
    
    // Preenche a matriz C com 0
    for (i = 0; i < matrixC->height * matrixC->width; i++) {
        matrixC->rows[i] = 0.0;
    }

    return matrixC;
}

static void saveMatrix(struct matrix * matrix, const char * fileName) {
    // TODO: Salvar matriz
}

// Libera espaço alocado para uma matriz
static void free_matrix(struct matrix * matrix) {
    free(matrix->rows);
    free(matrix);
}

int main(int argc, char* argv[]) {
    if (argc != 10) {
        printf("Parametros incorretos!\n");
        return 1;
    }

    /*
    Exemplo de comando:

    ./matrix_lib_test 5.0 8 16 16 8 floats_256_2.0f.dat floats_256_5.0f.dat result1.dat result2.dat

    5.0 é o valor escalar que multiplicará a primeira matriz;
    8 é o número de linhas da primeira matriz;
    16 é o número de colunas da primeira matriz;
    16 é o número de linhas da segunda matriz;
    8 é o número de colunas da segunda matriz;

    floats_256_2.0f.dat é o nome do arquivo de floats que será usado para carregar a primeira matriz;
    floats_256_5.0f.dat é o nome do arquivo de floats que será usado para carregar a segunda matriz;

    result1.dat é o nome do arquivo de floats onde o primeiro resultado será armazenado;
    result2.dat é o nome do arquivo de floats onde o segundo resultado será armazenado.
    */

    float scalar;
    unsigned long heightA, widthA, heightB, widthB;
    const char * floatsA, * floatsB, * result1, * result2;

    scalar = strtof(argv[1], NULL); // Conversão p/ float

    // Conversões p/ unsigned long, em base 10
    heightA = strtoul(argv[2], NULL, 10); 
    widthA = strtoul(argv[3], NULL, 10);
    heightB = strtoul(argv[4], NULL, 10);
    widthB = strtoul(argv[5], NULL, 10);

    floatsA = argv[6];
    floatsB = argv[7];

    result1 = argv[8];
    result1 = argv[9];

    // Inicialização das matrizes
    struct matrix * matrixA = init_matrix(heightA, widthA);
    struct matrix * matrixB = init_matrix(heightB, widthB);
    struct matrix * matrixC = init_matrixResult(matrixA, matrixB);

    // fillMatrix(matrixA, floatsA);
    // fillMatrix(matrixB, floatsB);
    
    /*
    A função scalar_matrix_mult deve ser chamada com os seguintes argumentos: o
    valor escalar fornecido e a primeira matriz (A). O resultado do produto
    (retornado na matriz A) deve ser armazenado em um arquivo binário usando o
    nome do terceiro arquivo de floats.
    */

    

    /*
    Depois, a função matrix_matrix_mult deve ser chamada com os seguintes
    argumentos: a matriz A resultante da função scalar_matrix_mult, a segunda
    matriz (B) e a terceira matriz (C). O resultado do produto (retornado na
    matriz C) deve ser armazenado em um arquivo binário com o nome do quarto
    arquivo de floats.
    */

    // Libera matrizes
    free_matrix(matrixA);
    free_matrix(matrixB);
    free_matrix(matrixC);

    return 0;
}