#include <stdint.h>
#include <stdio.h>
#include <veo_hmem.h>

uint64_t scalar_mult(unsigned long matrix_size, float * ve_rows, float scalar, int num_threads) {
    int tid, nthreads = 0;

    ve_rows = (float *) veo_get_hmem_addr(ve_rows);

    omp_set_num_threads(num_threads); // Limit threads for all consecutive parallel regions

    #pragma omp parallel private(nthreads, tid)
    {
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();

        if (tid == 0) {
            printf("scalar_mult: number of threads = %d\n", nthreads);
            fflush(stdout);
        }

        unsigned long i, num_elements_per_thread, extra_elements_for_last_thread, first_element_of_thread, limit_element_of_thread;

        num_elements_per_thread = matrix_size / nthreads;
        extra_elements_for_last_thread = matrix_size % nthreads;
        first_element_of_thread = tid * num_elements_per_thread;
        limit_element_of_thread = first_element_of_thread + num_elements_per_thread;

        if (tid == nthreads - 1) limit_element_of_thread += extra_elements_for_last_thread;

        for (i = first_element_of_thread; i < limit_element_of_thread; ++i) {
            ve_rows[i] *= scalar;
        }
    }

    return 0;
}

uint64_t matrix_mult(unsigned long matrix_size, float * ve_A, float * ve_B, float * ve_C, unsigned long widthA, unsigned long widthB, unsigned long widthC, int num_threads) {
    int tid, nthreads = 0;

    ve_A = (float *) veo_get_hmem_addr(ve_A);
    ve_B = (float *) veo_get_hmem_addr(ve_B);
    ve_C = (float *) veo_get_hmem_addr(ve_C);

    omp_set_num_threads(num_threads); // Limit threads for all consecutive parallel regions

    #pragma omp parallel private(nthreads, tid)
    {
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();

        if (tid == 0) {
            printf("matrix_mult: number of threads = %d\n", nthreads);
            fflush(stdout);
        }

        unsigned long i, num_elements_per_thread, extra_elements_for_last_thread, first_element_of_thread, limit_element_of_thread;

        num_elements_per_thread = matrix_size / nthreads;
        extra_elements_for_last_thread = matrix_size % nthreads;
        first_element_of_thread = tid * num_elements_per_thread;
        limit_element_of_thread = first_element_of_thread + num_elements_per_thread;

        if (tid == nthreads - 1) limit_element_of_thread += extra_elements_for_last_thread;

        unsigned long j, k, lineA, lineC, colC, endA, indexB;

        for (i = first_element_of_thread; i < limit_element_of_thread; ++i) {
            lineC = i / widthC;
            colC = i % widthC;
            lineA = lineC * widthA;
            endA = lineA + widthA;
            
            ve_C[i] = 0.0;
            
            for (j = lineA, k = 0; j < endA; j++, k++) {
                indexB = k * widthB + colC;
                ve_C[i] += ve_A[j] * ve_B[indexB];
            }
        }
    }

    return 0;
}
