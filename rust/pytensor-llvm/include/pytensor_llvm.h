/*
 * C interface to functions compiled by PyTensor's LLVM backend.
 *
 * A consumer (e.g. a sampler written in Rust or C) gets two things from Python:
 *
 *     api = pytensor_llvm.c_api_table()            # address of a PtFunctionApi
 *     fn  = compiled.c_function()                  # address of a PtFunction (new reference)
 *
 * Everything else goes through the function table, so nothing has to link against the
 * extension. Typical use from one sampler thread:
 *
 *     PtCallState *st = api->state_new(fn);        // per thread; owns reusable buffers
 *     for (...) {
 *         if (api->call(st, inputs, outputs) != 0)
 *             fprintf(stderr, "%s\n", api->last_error(st));
 *     }
 *     api->state_free(st);
 *     api->function_decref(fn);
 *
 * Threads: a PtFunction is immutable and may be shared; a PtCallState must only be used
 * by one thread at a time. No Python or GIL is involved in `call`.
 */
#ifndef PYTENSOR_LLVM_H
#define PYTENSOR_LLVM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PT_ABI_VERSION 1
#define PT_API_VERSION 1

/* dtype codes */
enum {
    PT_FLOAT64 = 0,
    PT_FLOAT32 = 1,
    PT_INT64 = 2,
    PT_INT32 = 3,
    PT_INT16 = 4,
    PT_INT8 = 5,
    PT_UINT8 = 6,
    PT_BOOL = 7, /* one byte, 0 or 1 */
};

typedef struct PtBuffer PtBuffer;       /* refcounted memory owned by the runtime */
typedef struct PtFunction PtFunction;   /* a compiled function, refcounted, immutable */
typedef struct PtCallState PtCallState; /* per-thread call state */

/* Same struct as in the kernel ABI. Strides are in bytes. */
typedef struct {
    char *data;
    int64_t *shape;
    int64_t *strides;
    PtBuffer *owner; /* NULL for memory owned by the caller */
    int64_t ndim;
} PtArray;

typedef struct {
    int32_t dtype;
    int32_t ndim;
    /* static length per dim, -1 if only known at runtime; valid while the
     * PtFunction is alive */
    const int64_t *static_shape;
} PtTypeInfo;

typedef struct {
    uint32_t abi_version; /* PT_ABI_VERSION */
    uint32_t api_version; /* PT_API_VERSION */

    void (*function_incref)(const PtFunction *fn);
    void (*function_decref)(const PtFunction *fn);
    size_t (*n_inputs)(const PtFunction *fn);
    size_t (*n_outputs)(const PtFunction *fn);
    /* 0 on success, nonzero if i is out of range */
    int32_t (*input_type)(const PtFunction *fn, size_t i, PtTypeInfo *out);
    int32_t (*output_type)(const PtFunction *fn, size_t i, PtTypeInfo *out);

    /* The state keeps a reference to fn. */
    PtCallState *(*state_new)(const PtFunction *fn);
    void (*state_free)(PtCallState *st);

    /*
     * Run the function. Returns 0 on success; on failure see last_error(st).
     *
     * inputs[i]: data, shape and strides of input i (ndim must match, dtype is not
     *   checked). The runtime never writes to inputs.
     *
     * outputs[k], chosen per output by the caller:
     *   - provided (data != NULL): shape must point to the expected shape (checked),
     *     strides to the strides to write with. The result is written there: directly by
     *     the producing kernel when the strides match what the planner chose (usually
     *     C-contiguous), otherwise via one copy. owner is left untouched.
     *   - runtime-allocated (data == NULL): shape and strides must point to caller
     *     arrays of length ndim; the runtime fills them, sets data, and sets owner to a
     *     new reference the caller releases with buffer_decref.
     */
    int32_t (*call)(PtCallState *st, const PtArray *inputs, PtArray *outputs);

    /* Message of the last failed call on this state (valid until the next call). */
    const char *(*last_error)(const PtCallState *st);

    void (*buffer_decref)(PtBuffer *buf);
} PtFunctionApi;

#ifdef __cplusplus
}
#endif

#endif /* PYTENSOR_LLVM_H */
