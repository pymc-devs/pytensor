/*
 * macOS sdot_ bug workaround.
 *
 * Apple's Accelerate framework has a bug where the Fortran sdot_() interface
 * returns incorrect values. This wrapper uses cblas_sdot() instead, which
 * works correctly.
 */

extern "C" float cblas_sdot(int, float*, int, float*, int);
static float sdot_(int* Nx, float* x, int* Sx, float* y, int* Sy)
{
    return cblas_sdot(*Nx, x, *Sx, y, *Sy);
}

