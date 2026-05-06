/*
 * MKL threads interface declarations for PyTensor.
 */

#ifndef PYTENSOR_MKL_THREADS_H
#define PYTENSOR_MKL_THREADS_H

extern "C"
{
    int     MKL_Set_Num_Threads_Local(int);
    #define mkl_set_num_threads_local   MKL_Set_Num_Threads_Local

    void    MKL_Set_Num_Threads(int);
    #define mkl_set_num_threads         MKL_Set_Num_Threads

    int     MKL_Get_Max_Threads(void);
    #define mkl_get_max_threads         MKL_Get_Max_Threads

    int     MKL_Domain_Set_Num_Threads(int, int);
    #define mkl_domain_set_num_threads  MKL_Domain_Set_Num_Threads

    int     MKL_Domain_Get_Max_Threads(int);
    #define mkl_domain_get_max_threads  MKL_Domain_Get_Max_Threads

    void    MKL_Set_Dynamic(int);
    #define mkl_set_dynamic             MKL_Set_Dynamic

    int     MKL_Get_Dynamic(void);
    #define mkl_get_dynamic             MKL_Get_Dynamic
}

#endif /* PYTENSOR_MKL_THREADS_H */

