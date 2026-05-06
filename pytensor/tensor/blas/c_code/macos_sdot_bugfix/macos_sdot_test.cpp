/*
 * Test program to detect the macOS BLAS sdot_ bug.
 *
 * Apple's Accelerate framework has a long-standing bug where the Fortran
 * interface sdot_() returns incorrect values. This test computes a simple
 * dot product and checks if the result is correct.
 *
 * Expected result: 0*0 + 1*1 + 2*2 + 3*3 + 4*4 = 30
 * Returns 0 if correct, -1 if bug is present.
 */

extern "C" float sdot_(int*, float*, int*, float*, int*);

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

