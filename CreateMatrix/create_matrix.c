#include <stdio.h>
#include <stdlib.h>

static void saveMatrix(unsigned long tam, float num, const char * fileName) {
    FILE *file;
    
    file = fopen(fileName, "wb");

    if (file == NULL) {
        printf("Erro na abertura do arquivo.\n");
        system("pause");
        exit(1);
    }

    int i;
    float * v = (float *) malloc(sizeof(float) * tam);
    for (i = 0; i < tam; i++) {
        v[i] = num;
    }

    fwrite(v, tam * sizeof(float), 1, file);
    fclose(file);
    free(v);
}

static void readMatrix(unsigned long tam, const char * fileName, int print) {
    FILE *file;
    
    file = fopen(fileName, "rb");

    if (file == NULL) {
        printf("Erro na abertura do arquivo.\n");
        system("pause");
        exit(1);
    }

    float * v = (float *) malloc(sizeof(float) * tam);

    fread(v, tam * sizeof(float), 1, file);

    if (print) {
        int i;
        for (i = 0; i < tam; i++) {
            printf("%f ", v[i]);
        }
        printf("\n");
    }

    fclose(file);
    free(v);
}

int main(int argc, char* argv[]) {
    // Argumentos padrao, caso sejam passados incorretamente
    unsigned long numFloats = 64;
    float numInserido = 1.0;
    const char * arquivo = "floats_64_1.0.dat";
    int print = 0;

    if (argc == 6) {
        // ./create_matrix <num> <altura> <largura> <arquivo> <print>
        
        numInserido = strtof(argv[1], NULL);
        numFloats = strtoul(argv[2], NULL, 10) * strtoul(argv[3], NULL, 10);
        arquivo = argv[4];
        print = atoi(argv[5]);
    } else {
        printf("Numero incorreto de argumentos.\n");
        printf("Uso correto: ./create_matrix <num> <altura> <largura> <arquivo> <print>\n");
    }

    printf("Criando arquivo de contendo %d floats com valor %.1f...\n", numFloats, numInserido);

    saveMatrix(numFloats, numInserido, arquivo);
    printf("Teste arquivo %s\n", arquivo);
    readMatrix(numFloats, arquivo, print);
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

    return 0;
}