/*
 * Test program to verify the macOS BLAS sdot_ bug workaround.
 *
 * This defines a static sdot_ wrapper that uses cblas_sdot internally,
 * then tests if it returns the correct result. The C interface cblas_sdot()
 * works correctly even when the Fortran sdot_() is buggy.
 *
 * Expected result: 0*0 + 1*1 + 2*2 + 3*3 + 4*4 = 30
 * Returns 0 if workaround works, -1 if it fails.
 */

extern "C" float cblas_sdot(int, float*, int, float*, int);

static float sdot_(int* Nx, float* x, int* Sx, float* y, int* Sy)
{
    return cblas_sdot(*Nx, x, *Sx, y, *Sy);
}

int main(int argc, char** argv)
{
    int Nx = 5;
    int Sx = 1;
    float x[5] = {0, 1, 2, 3, 4};
    float r = sdot_(&Nx, x, &Sx, x, &Sx);

    if ((r - 30.f) > 1e-6 || (r - 30.f) < -1e-6)
    {
        return -1;
    }
    return 0;
}

