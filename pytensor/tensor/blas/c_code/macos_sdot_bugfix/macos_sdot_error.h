/*
 * macOS sdot_ bug fatal error stub.
 *
 * When the sdot_ bug is detected but no workaround is available,
 * this stub ensures we fail loudly rather than silently returning
 * incorrect results.
 */

static float sdot_(int* Nx, float* x, int* Sx, float* y, int* Sy)
{
    fprintf(stderr,
        "FATAL: The implementation of BLAS SDOT "
        "routine in your system has a bug that "
        "makes it return wrong results.\n"
        "You can work around this bug by using a "
        "different BLAS library, or disabling BLAS\n");
    assert(0);
}

