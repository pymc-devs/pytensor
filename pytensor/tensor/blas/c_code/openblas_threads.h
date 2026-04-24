/*
 * OpenBLAS threads interface declarations for PyTensor.
 */

#ifndef PYTENSOR_OPENBLAS_THREADS_H
#define PYTENSOR_OPENBLAS_THREADS_H

extern "C"
{
    void openblas_set_num_threads(int);
    void goto_set_num_threads(int);
    int openblas_get_num_threads(void);
}

#endif /* PYTENSOR_OPENBLAS_THREADS_H */

