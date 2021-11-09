#include<stdlib.h>
#include <stdio.h>
#pragma warning(disable : 4996)

int main(void) {


    FILE* file;
    int i = 0;
    float binario = 1;

    file = fopen("Matrix_Test_1.dat", "wb");

    if (file == NULL) {
        printf("Erro na abertura do arquivo.\n");
        system("pause");
        exit(1);
    }


    while (i < 4) {
        fwrite(&binario, sizeof(binario), 1, file);
        i++;

    }
    fclose(file);

    return 0;
}