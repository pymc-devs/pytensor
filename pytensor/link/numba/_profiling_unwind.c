#include <elfutils/libdwfl.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    Dwfl *dwfl;
    uint64_t regs[17];
    uint64_t stack_addr;
    const unsigned char *stack;
    size_t stack_size;
    uint64_t *pcs;
    int n, max;
} Context;

static pid_t next_thread(Dwfl *dwfl, void *arg, void **thread_arg) {
    if (*thread_arg) return 0;
    *thread_arg = arg;
    return 1;
}
static bool get_thread(Dwfl *dwfl, pid_t tid, void *arg, void **thread_arg) {
    *thread_arg = arg;
    return tid == 1;
}
static bool memory_read(Dwfl *dwfl, Dwarf_Addr addr, Dwarf_Word *result, void *arg) {
    Context *ctx = arg;
    if (ctx->stack_size < 8 || addr < ctx->stack_addr ||
        addr - ctx->stack_addr > ctx->stack_size - 8) return false;
    memcpy(result, ctx->stack + addr - ctx->stack_addr, 8);
    return true;
}
static bool registers(Dwfl_Thread *thread, void *arg) {
    Context *ctx = arg;
    dwfl_thread_state_register_pc(thread, ctx->regs[16]);
    return dwfl_thread_state_registers(thread, 0, 17, ctx->regs);
}
static const Dwfl_Thread_Callbacks threads = {
    .next_thread = next_thread, .get_thread = get_thread,
    .memory_read = memory_read, .set_initial_registers = registers,
};
static const Dwfl_Callbacks callbacks = {
    .find_debuginfo = dwfl_standard_find_debuginfo,
};

void *probe_create(void) {
    Context *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    ctx->dwfl = dwfl_begin(&callbacks);
    if (!ctx->dwfl) { free(ctx); return NULL; }
    return ctx;
}
int probe_module(Context *ctx, const char *path, uint64_t base) {
    return dwfl_report_elf(ctx->dwfl, path, path, -1, base, false) != NULL;
}
int probe_attach(Context *ctx) {
    if (dwfl_report_end(ctx->dwfl, NULL, NULL)) return 0;
    return dwfl_attach_state(ctx->dwfl, NULL, 1, &threads, ctx);
}
static int frame(Dwfl_Frame *state, void *arg) {
    Context *ctx = arg;
    if (ctx->n >= ctx->max) return DWARF_CB_ABORT;
    Dwarf_Addr pc;
    bool activation;
    if (!dwfl_frame_pc(state, &pc, &activation) || !pc) return DWARF_CB_ABORT;
    ctx->pcs[ctx->n++] = pc;
    if (ctx->n >= ctx->max || !dwfl_addrmodule(ctx->dwfl, pc - !activation))
        return DWARF_CB_ABORT;
    return DWARF_CB_OK;
}
int probe_unwind(Context *ctx, const uint64_t *regs, uint64_t stack_addr,
                 const void *stack, size_t stack_size, uint64_t *pcs, int max) {
    memcpy(ctx->regs, regs, sizeof(ctx->regs));
    ctx->stack_addr = stack_addr;
    ctx->stack = stack;
    ctx->stack_size = stack_size;
    ctx->pcs = pcs;
    ctx->n = 0;
    ctx->max = max;
    dwfl_getthread_frames(ctx->dwfl, 1, frame, ctx);
    return ctx->n;
}
const char *probe_name(Context *ctx, uint64_t pc) {
    Dwfl_Module *mod = dwfl_addrmodule(ctx->dwfl, pc);
    return mod ? dwfl_module_addrname(mod, pc) : NULL;
}
const char *probe_path(Context *ctx, uint64_t pc) {
    Dwfl_Module *mod = dwfl_addrmodule(ctx->dwfl, pc);
    return mod ? dwfl_module_info(mod, NULL, NULL, NULL, NULL, NULL, NULL, NULL) : NULL;
}
const char *probe_error(void) { return dwfl_errmsg(-1); }
void probe_close(Context *ctx) { dwfl_end(ctx->dwfl); free(ctx); }
int probe_symbol(Context *ctx, uint64_t pc, uint64_t *out) {
    Dwfl_Module *mod = dwfl_addrmodule(ctx->dwfl, pc);
    GElf_Off offset;
    GElf_Sym sym;
    if (!mod || !dwfl_module_addrinfo(mod, pc, &offset, &sym, NULL, NULL, NULL)) return 0;
    out[0] = pc - offset;
    out[1] = sym.st_size;
    out[2] = sym.st_value;
    return sym.st_size > 0 && offset < sym.st_size;
}
