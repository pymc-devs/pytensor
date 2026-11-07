/* A C consumer of the PyTensor LLVM C API, used by test_c_api.py.
 * Only depends on pytensor_llvm.h; gets the API table and function from Python. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pytensor_llvm.h"

#define MAXD 32

static size_t itemsize(int32_t dt) {
    switch (dt) {
    case PT_FLOAT64: case PT_INT64: return 8;
    case PT_FLOAT32: case PT_INT32: return 4;
    case PT_INT16: return 2;
    default: return 1;
    }
}

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

/* One call; outputs as given by the caller (provided or data == NULL). */
int pt_run_once(const PtFunctionApi *api, const PtFunction *fn, const PtArray *inputs,
                PtArray *outputs, char *err, size_t errlen) {
    PtCallState *st = api->state_new(fn);
    int rc = api->call(st, inputs, outputs);
    if (rc != 0) snprintf(err, errlen, "%s", api->last_error(st));
    api->state_free(st);
    return rc;
}

/* Average ns per call over `iters` calls on one state.
 * provided != 0: output buffers are allocated once here (shapes from a first call) and
 *                reused, as a sampler would do.
 * provided == 0: the runtime allocates outputs on every call; we release them. */
double pt_bench(const PtFunctionApi *api, const PtFunction *fn, const PtArray *inputs,
                int provided, long iters, char *err, size_t errlen) {
    if (api->abi_version != PT_ABI_VERSION) {
        snprintf(err, errlen, "ABI version mismatch");
        return -1;
    }
    size_t nout = api->n_outputs(fn);
    PtArray *outs = calloc(nout, sizeof(PtArray));
    int64_t *shapes = calloc(nout * MAXD, sizeof(int64_t));
    int64_t *strides = calloc(nout * MAXD, sizeof(int64_t));
    void **bufs = calloc(nout, sizeof(void *));
    PtCallState *st = api->state_new(fn);
    double result = -1;

    for (size_t k = 0; k < nout; k++) {
        PtTypeInfo t;
        api->output_type(fn, k, &t);
        outs[k].ndim = t.ndim;
        outs[k].shape = shapes + k * MAXD;
        outs[k].strides = strides + k * MAXD;
    }
    /* first call with runtime-allocated outputs tells us the output shapes */
    if (api->call(st, inputs, outs) != 0) {
        snprintf(err, errlen, "%s", api->last_error(st));
        goto done;
    }
    for (size_t k = 0; k < nout; k++) {
        api->buffer_decref(outs[k].owner);
        outs[k].owner = NULL;
        outs[k].data = NULL;
        if (provided) {
            PtTypeInfo t;
            api->output_type(fn, k, &t);
            size_t isz = itemsize(t.dtype), n = 1;
            for (int d = (int)outs[k].ndim - 1; d >= 0; d--) {
                outs[k].strides[d] = (int64_t)(n * isz); /* C-contiguous */
                n *= (size_t)outs[k].shape[d];
            }
            bufs[k] = aligned_alloc(64, ((n * isz + 63) / 64) * 64 + 64);
            outs[k].data = bufs[k];
        }
    }
    for (int warm = 0; warm < 2; warm++) {
        double t0 = now_ns();
        long n = warm ? iters : iters / 10 + 1;
        for (long i = 0; i < n; i++) {
            if (api->call(st, inputs, outs) != 0) {
                snprintf(err, errlen, "%s", api->last_error(st));
                goto done;
            }
            if (!provided) {
                for (size_t k = 0; k < nout; k++) {
                    api->buffer_decref(outs[k].owner);
                    outs[k].owner = NULL;
                    outs[k].data = NULL;
                }
            }
        }
        result = (now_ns() - t0) / (double)n;
    }
done:
    api->state_free(st);
    for (size_t k = 0; k < nout; k++) free(bufs[k]);
    free(bufs);
    free(outs);
    free(shapes);
    free(strides);
    return result;
}
