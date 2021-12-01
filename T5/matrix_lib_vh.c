#include <stdlib.h>
#include <stdio.h>
#include <ve_offload.h>
#include "matrix_lib.h"


/*
NEC Vector Engine - T5
*/
static int ve_execution_node = 0;
static int nThreads = 1;
static struct veo_proc_handle * process = NULL;
static uint64_t library_handle = 0;

void set_ve_execution_node(int num_node) {
    if (num_node < 0 || num_node > 3) {
        ve_execution_node = 0;
        printf("num_node %d is out of range. Using default value\n", num_node);
        return;
    }

    ve_execution_node = num_node;
}

void set_number_threads(int num_threads) {
    if (num_threads < 1) {
        nThreads = 1;
        printf("num_threads %d is invalid. Using 1 thread\n", num_threads);
        return;
    }

    nThreads = num_threads <= 8 ? num_threads : 8;
}

int init_proc_ve_node() {
    process = veo_proc_create(ve_execution_node);
    if (process == NULL) {
        printf("veo_proc_create failed\n");
        return 0;
    }

    library_handle = veo_load_library(process, "./matrix_lib_ve.so");

    if (library_handle == 0) {
        printf("veo_load_library failed\n");
        veo_proc_destroy(process);
        process = NULL;
        return 0;
    }
    return 1;
}

int close_proc_ve_node() {
    int rc = veo_unload_library(process, library_handle);
    if (rc != 0) {
        printf("veo_unload_library failed: %d\n", rc);
        return 0;
    }
    library_handle = 0;
    rc = veo_proc_destroy(process);
    if (rc != 0) {
        printf("veo_proc_destroy failed: %d\n", rc);
        return 0;
    }
    process = NULL;
    return 1;
}

int load_ve_matrix(struct matrix *matrix) {
    int ret = veo_alloc_hmem(process, &(matrix->ve_rows), sizeof(float) * matrix->height * matrix->width);
    if (ret != 0) {
        printf("veo_alloc_hmem failed for ve_rows: %d\n", ret);
        return 0;
    }

    return sync_vh_ve_matrix(matrix);
}

int unload_ve_matrix(struct matrix *matrix) {
    if (sync_ve_vh_matrix(matrix) == 0) {
        printf("unload_ve_matrix failed when copying back\n");
        return 0;
    }

    int ret = veo_free_hmem(matrix->ve_rows);
    if (ret != 0) {
        printf("veo_free_hmem failed: %d\n", ret);
        return 0;
    }
    matrix->ve_rows = NULL;

    return 1;
}

/// Returns 0 when there is an error: either vh_rows or ve_rows are NULL
static int verify_rows_for_null(Matrix * matrix) {
    if (matrix->vh_rows == NULL) {
        printf("vh_rows NULL!!!!\n");
        return 0;
    }
    if (matrix->ve_rows == NULL) {
        printf("ve_rows NULL!!!!\n");
        return 0;
    }
    return 1;
}

int sync_vh_ve_matrix(struct matrix *matrix) {
    if (verify_rows_for_null(matrix) == 0) {
        printf("Error: found NULL when trying to copy (vh_rows -> ve_rows)\n");
        return 0;
    }

    int ret = veo_hmemcpy(matrix->ve_rows, matrix->vh_rows, sizeof(float) * matrix->height * matrix->width);
    if (ret != 0) {
        printf("veo_hmemcpy (vh_rows -> ve_rows) failed: %d\n", ret);
        return 0;
    }
    return 1;
}

int sync_ve_vh_matrix(struct matrix *matrix) {
    if (verify_rows_for_null(matrix) == 0) {
        printf("Error: found NULL when trying to copy (ve_rows -> vh_rows)\n");
        return 0;
    }

    int ret = veo_hmemcpy(matrix->vh_rows, matrix->ve_rows, sizeof(float) * matrix->height * matrix->width);
    if (ret != 0) {
        printf("veo_hmemcpy (ve_rows -> vh_rows) failed: %d\n", ret);
        return 0;
    }
    return 1;
}


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

    struct veo_thr_ctxt *ctx = veo_context_open(process);
    if (ctx == NULL) {
        printf("veo_context_open (scalar mult) failed\n");
        return 0;
    }

    // Preparing arguments
    int ret;
    struct veo_args *argp = veo_args_alloc();
    if (argp == NULL) {
        printf("veo_args_alloc (scalar mult) failed\n");
        return 0;
    }
    veo_args_clear(argp);

    // matrix_size
    ret = veo_args_set_u64(argp, 0, matrix->height * matrix->width);
    if (ret != 0) {
        printf("veo_args_set_u64 failed for matrix_size: %d", ret);
        return 0;
    }

    // ve_rows
    ret = veo_args_set_hmem(argp, 1, matrix->ve_rows);
    if (ret != 0) {
        printf("veo_args_set_hmem failed for ve_rows: %d", ret);
        return 0;
    }

    // scalar
    ret = veo_args_set_float(argp, 2, scalar_value);
    if (ret != 0) {
        printf("veo_args_set_hmem failed for scalar: %d", ret);
        return 0;
    }

    // num_threads
    ret = veo_args_set_i32(argp, 3, nThreads);
    if (ret != 0) {
        printf("veo_args_set_hmem failed for num_threads: %d", ret);
        return 0;
    }

    // Running threads on VE
    uint64_t id = veo_call_async_by_name(ctx, library_handle, "scalar_mult", argp);
    if (id == VEO_REQUEST_ID_INVALID) {
        printf("veo_call_async_by_name (scalar mult) failed: %lu\n", id);
        return 0;
    }

    // Waiting for threads to finish
    uint64_t retval;
    int wait_val;
    wait_val = veo_call_wait_result(ctx, id, &retval);
    if (wait_val != VEO_COMMAND_OK) {
        printf("veo_call_wait_result (scalar mult) failed: %d\n", wait_val);
        return 0;
    }

    // Closing
    veo_args_free(argp);
    ret = veo_context_close(ctx);
    if (ret != 0) {
        printf("veo_context_close (scalar mult) failed: %d\n", ret);
        return 0;
    }

    return 1;
}


// void matrix_thread(unsigned long n, float * d_A, float * d_B, float * d_C, unsigned long widthA, unsigned long widthB, unsigned long widthC) {
//     unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     unsigned long j, k, lineA, lineC, colC, endA, indexB;

//     for (; i < n; i += stride) {
//         lineC = i / widthC;
//         colC = i % widthC;
//         lineA = lineC * widthA;
//         endA = lineA + widthA;
        
//         d_C[i] = 0.0;
        
//         for (j = lineA, k = 0; j < endA; j++, k++) {
//             indexB = k * widthB + colC;
//             d_C[i] += d_A[j] * d_B[indexB];
//         }
//     }
// }

int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC) {
    if (test_matrix(matrixA) == 0 || test_matrix(matrixB) == 0) // testa a matrix
        return 0;

    // calcula o produto de uma matriz A (m x n) por uma matriz B (n x p),
    // armazenando o resultado na matriz C (m x p), previamente criada

    // testa se é possivel fazer a multiplicação entre as matrizes (m x n * n x p)
    if (matrixA->width != matrixB->height)
        return 0;

    if (matrixA->height != matrixC->height || matrixB->width != matrixC->width) 
        return 0;

    struct veo_thr_ctxt *ctx = veo_context_open(process);
    if (ctx == NULL) {
        printf("veo_context_open (matrix mult) failed\n");
        return 0;
    }

    // Preparing arguments
    int ret;
    struct veo_args *argp = veo_args_alloc();
    if (argp == NULL) {
        printf("veo_args_alloc (matrix mult) failed\n");
        return 0;
    }
    veo_args_clear(argp);

    // matrix_size
    ret = veo_args_set_u64(argp, 0, matrixC->height * matrixC->width);
    if (ret != 0) {
        printf("veo_args_set_u64 failed for height: %d", ret);
        return 0;
    }

    // ve_rows
    ret = veo_args_set_hmem(argp, 1, matrixC->ve_rows);
    if (ret != 0) {
        printf("veo_args_set_hmem failed for ve_rows: %d", ret);
        return 0;
    }

    // scalar
    ret = veo_args_set_float(argp, 2, 1.0);
    if (ret != 0) {
        printf("veo_args_set_hmem failed for scalar: %d", ret);
        return 0;
    }

    // num_threads
    ret = veo_args_set_i32(argp, 3, nThreads);
    if (ret != 0) {
        printf("veo_args_set_hmem failed for num_threads: %d", ret);
        return 0;
    }

    // Running threads on VE
    uint64_t id = veo_call_async_by_name(ctx, library_handle, "matrix_mult", argp);
    if (id == VEO_REQUEST_ID_INVALID) {
        printf("veo_call_async_by_name (matrix mult) failed: %lu\n", id);
        return 0;
    }

    // int blockSize = threadsPerBlock, matrixSize = matrixC->height * matrixC->width;
    // int numBlocks = (matrixSize + blockSize - 1) / blockSize;
    // if (numBlocks > maxBlocksPerGrid)
    //     numBlocks = maxBlocksPerGrid;

    // matrix_thread<<<numBlocks, blockSize>>>(matrixSize, matrixA->d_rows, matrixB->d_rows, matrixC->d_rows, matrixA->width, matrixB->width, matrixC->width);

    // Waiting for threads to finish
    uint64_t retval;
    int wait_val;
    wait_val = veo_call_wait_result(ctx, id, &retval);
    if (wait_val != VEO_COMMAND_OK) {
        printf("veo_call_wait_result (matrix mult) failed: %d\n", wait_val);
        return 0;
    }

    // Closing
    veo_args_free(argp);
    ret = veo_context_close(ctx);
    if (ret != 0) {
        printf("veo_context_close (matrix mult) failed: %d\n", ret);
        return 0;
    }

    return 1;
}
