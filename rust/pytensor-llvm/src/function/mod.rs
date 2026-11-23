//! A compiled function: the generated kernel of a whole graph (Python's `driver.py`),
//! with the types of its arguments.
//!
//! The kernel has the kernel ABI. Every graph input is an input; every graph output is
//! an output, provided (static shapes: C-contiguous memory the caller gives) or returned
//! (the kernel hands over one reference to a buffer, never a view). Everything else
//! (which kernels run, where values live, when they are released) is in the generated
//! code.
//!
//! This is where calls from outside (Python, the C API) meet it: their arrays are
//! checked against the types here, and outputs land where the caller wants them: in
//! memory it provides (written directly when the kernel can, else copied there) or in
//! new buffers. Calls run in a `Session`, which keeps what they need between them, so
//! that calling a function with static shapes allocates nothing.

pub mod capi;

use crate::abi::{KernelFn, OwnedBuf, PtArray, c_strides, checked_nbytes, overlaps};
use crate::array::{OwnedArray, PtArrayMut, PtArrayRef};
use crate::dtype::DType;
use crate::runtime::parallel::CallerWaits;
use crate::runtime::{PtCtx, runtime};

pub type R<T> = Result<T, String>;

/// The type of an argument: dtype and static lengths (None where only known at run
/// time).
#[derive(Clone, Debug)]
pub struct ArgType {
    pub dtype: DType,
    pub shape: Vec<Option<u64>>,
}

impl ArgType {
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn static_shape(&self) -> Option<Vec<i64>> {
        self.shape.iter().map(|s| s.map(|n| n as i64)).collect()
    }

    /// Static lengths, -1 where only known at run time.
    pub fn shape_or_unknown(&self) -> Vec<i64> {
        self.shape
            .iter()
            .map(|s| s.map_or(-1, |n| n as i64))
            .collect()
    }

    fn check(&self, shape: &[i64]) -> R<()> {
        if shape.len() != self.ndim() {
            return Err(format!(
                "expected {} axes, got {}",
                self.ndim(),
                shape.len()
            ));
        }
        for (a, (&n, s)) in shape.iter().zip(&self.shape).enumerate() {
            if n < 0 {
                return Err(format!("negative length on axis {a}"));
            }
            if let Some(s) = s
                && n as u64 != *s
            {
                return Err(format!("length {n} on axis {a}, the type says {s}"));
            }
        }
        Ok(())
    }
}

/// Where the caller wants an output.
pub enum Destination<'c> {
    /// in a new buffer
    New,
    /// in this memory
    Into(PtArrayMut<'c>),
}

pub struct Function {
    kernel: KernelFn,
    config: *const u8,
    pub inputs: Vec<ArgType>,
    /// inputs the generated code assumes C-contiguous (Python calls copy other arrays,
    /// the C API rejects them)
    pub contiguous_inputs: Vec<bool>,
    pub outputs: Vec<ArgType>,
    /// per output: the kernel writes it into provided memory (else returns a buffer)
    provided: Vec<bool>,
    /// per provided output: its static shape, then its C-contiguous strides (the
    /// header the kernel gets; empty for returned outputs)
    provided_headers: Vec<Vec<i64>>,
    /// per provided output: its size in bytes
    provided_bytes: Vec<usize>,
    /// how many i64s the shapes and strides of the returned outputs take
    returned_header_size: usize,
    /// may the kernel run work on the runtime's thread pool (`runs_on_pool`)?
    uses_pool: bool,
}

// SAFETY: the kernel and its config are usable from any thread (`Function::new`'s
// contract); the rest is plain data.
unsafe impl Send for Function {}
// SAFETY: as above.
unsafe impl Sync for Function {}

impl Function {
    /// # Safety
    /// `address` is a function with the kernel ABI and `config` the config it expects,
    /// both valid as long as the function is used, from any thread. It computes
    /// outputs of the types `outputs` from inputs of the types `inputs`, writing the
    /// `provided` ones into the C-contiguous memory it gets and returning a buffer
    /// reference for the others (the kernel contract). `uses_pool`: may it run work
    /// on the runtime's thread pool?
    pub unsafe fn new(
        address: usize,
        config: usize,
        inputs: Vec<ArgType>,
        contiguous_inputs: Vec<bool>,
        outputs: Vec<ArgType>,
        provided: Vec<bool>,
        uses_pool: bool,
    ) -> R<Function> {
        // SAFETY: a null or a function pointer of this type (the function's contract).
        let kernel = unsafe { std::mem::transmute::<usize, Option<KernelFn>>(address) }
            .ok_or("null function")?;
        if contiguous_inputs.len() != inputs.len() || provided.len() != outputs.len() {
            return Err("one flag per argument".into());
        }
        let mut provided_headers = vec![];
        let mut provided_bytes = vec![];
        let mut returned_header_size = 0;
        for (o, t) in outputs.iter().enumerate() {
            if !provided[o] {
                provided_headers.push(vec![]);
                provided_bytes.push(0);
                returned_header_size += 2 * t.ndim();
                continue;
            }
            let Some(shape) = t.static_shape() else {
                return Err(format!("output {o}: provided, but its shape isn't static"));
            };
            let itemsize = t.dtype.itemsize();
            provided_bytes.push(
                checked_nbytes(&shape, itemsize)
                    .ok_or_else(|| crate::abi::bad_size("output", &shape))?,
            );
            let strides = c_strides(&shape, itemsize as i64);
            provided_headers.push([shape, strides].concat());
        }
        Ok(Function {
            kernel,
            config: config as *const u8,
            inputs,
            contiguous_inputs,
            outputs,
            provided,
            provided_headers,
            provided_bytes,
            returned_header_size,
            uses_pool,
        })
    }

    /// Does a call in a session with the context `ctx` run on the runtime's thread
    /// pool? It does when the function may use the pool, and the session allows
    /// parallelism. The whole call then runs on one of the pool's threads, so that its
    /// parallel parts start there and only what other threads steal leaves it, instead
    /// of each of them hopping over to the pool and back (rayon runs work on the
    /// calling thread only when that is one of its own).
    pub fn runs_on_pool(&self, ctx: &PtCtx) -> bool {
        self.uses_pool && ctx.parallel && runtime().threads() > 1
    }

    pub fn n_inputs(&self) -> usize {
        self.inputs.len()
    }

    pub fn n_outputs(&self) -> usize {
        self.outputs.len()
    }

    /// Check input `k`: its dtype, its shape against the type, and that it is
    /// C-contiguous if the function was compiled for that.
    pub fn check_input(&self, k: usize, a: &PtArrayRef) -> R<()> {
        let t = &self.inputs[k];
        if a.dtype() != t.dtype {
            return Err(format!(
                "input {k}: an array of {} is expected, got one of {}",
                t.dtype.numpy_name(),
                a.dtype().numpy_name()
            ));
        }
        t.check(a.shape()).map_err(|m| format!("input {k}: {m}"))?;
        if self.contiguous_inputs[k] && !a.is_c_contiguous() {
            return Err(format!(
                "input {k}: must be C-contiguous (the function was compiled for C-contiguous inputs)"
            ));
        }
        Ok(())
    }

    /// Run the function in `session`, with `inputs` and the destinations of the
    /// outputs (`outputs`, one per output). Inputs are checked against their types
    /// (`check_input`), memory for outputs against the output types and against
    /// overlapping the inputs and each other.
    ///
    /// Returns the outputs whose destination is `Destination::New`: (output, array),
    /// in output order. Those the caller doesn't take are released. On an error,
    /// nothing is returned (and nothing is kept).
    pub fn call<'s, 'c>(
        &self,
        session: &'s mut Session,
        inputs: impl IntoIterator<Item = R<PtArrayRef<'c>>>,
        outputs: impl IntoIterator<Item = R<Destination<'c>>>,
    ) -> R<std::vec::Drain<'s, (usize, OwnedArray)>> {
        let s = session;
        s.args.clear();
        s.targets.clear();
        s.in_ranges.clear();
        s.out_ranges.clear();
        s.results.clear();
        for a in inputs {
            let k = s.args.len();
            if k == self.n_inputs() {
                return Err(format!("expected {} inputs", self.n_inputs()));
            }
            let a = a?;
            self.check_input(k, &a)?;
            s.in_ranges.push(a.extent());
            s.args.push(a.to_abi());
        }
        for d in outputs {
            let o = s.targets.len();
            if s.args.len() != self.n_inputs() || o == self.n_outputs() {
                return Err(format!(
                    "expected {} inputs and {} outputs",
                    self.n_inputs(),
                    self.n_outputs()
                ));
            }
            let target = self.target(s, o, d?)?;
            s.targets.push(target);
        }
        if s.args.len() != self.n_inputs() || s.targets.len() != self.n_outputs() {
            return Err(format!(
                "expected {} inputs and {} outputs, got {} and {}",
                self.n_inputs(),
                self.n_outputs(),
                s.args.len(),
                s.targets.len()
            ));
        }
        // SAFETY: the arguments of this call, checked above; the arrays they describe
        // are borrowed for `'c`, which lasts until we return.
        unsafe { self.run(s) }?;
        Ok(s.results.drain(..))
    }

    /// Where output `o` goes, for `destination`.
    fn target(&self, s: &mut Session, o: usize, destination: Destination) -> R<Target> {
        let t = &self.outputs[o];
        let Destination::Into(m) = destination else {
            return Ok(Target {
                array: PtArray {
                    data: std::ptr::null_mut(),
                    shape: std::ptr::null_mut(),
                    strides: std::ptr::null_mut(),
                    owner: std::ptr::null_mut(),
                    ndim: t.ndim() as i32,
                    dtype: t.dtype.code(),
                },
                direct: false,
            });
        };
        if m.dtype() != t.dtype {
            return Err(format!(
                "provided output {o}: an array of {} is expected, got one of {}",
                t.dtype.numpy_name(),
                m.dtype().numpy_name()
            ));
        }
        t.check(m.shape())
            .map_err(|e| format!("provided output {o}: {e}"))?;
        let r = m.extent();
        if let Some(i) = s.in_ranges.iter().position(|&x| overlaps(x, r)) {
            return Err(format!(
                "provided output {o} overlaps the memory of input {i}"
            ));
        }
        if s.out_ranges.iter().any(|&x| overlaps(x, r)) {
            return Err(format!(
                "provided output {o} overlaps another provided output"
            ));
        }
        s.out_ranges.push(r);
        Ok(Target {
            direct: self.provided[o] && m.is_c_contiguous(),
            array: m.into_abi(),
        })
    }

    /// Run the kernel with the inputs in `s.args` and the outputs `s.targets`, and put
    /// the new arrays into `s.results`.
    ///
    /// # Safety
    /// `s.args` holds the inputs and `s.targets` the outputs of a call (`call`), all of
    /// them checked, and the arrays they describe are borrowed until we return.
    unsafe fn run(&self, s: &mut Session) -> R<()> {
        let (n_in, n_out) = (self.n_inputs(), self.n_outputs());
        s.headers.clear();
        s.headers.resize(self.returned_header_size, 0);
        let headers = s.headers.as_mut_ptr();
        let mut at = 0;
        s.staged.clear();
        for (o, t) in self.outputs.iter().enumerate() {
            let nd = t.ndim();
            let mut arg = PtArray {
                data: std::ptr::null_mut(),
                shape: std::ptr::null_mut(),
                strides: std::ptr::null_mut(),
                owner: std::ptr::null_mut(),
                ndim: nd as i32,
                dtype: t.dtype.code(),
            };
            let mut staged = None;
            if self.provided[o] {
                // (kernels don't write the headers of provided outputs)
                let header = self.provided_headers[o].as_ptr() as *mut i64;
                arg.shape = header;
                // SAFETY: the header has 2 * nd entries.
                arg.strides = unsafe { header.add(nd) };
                let target = &s.targets[o];
                arg.data = if target.direct {
                    target.array.data
                } else {
                    let n = self.provided_bytes[o];
                    let buf = s
                        .ctx
                        .alloc(n)
                        .ok_or_else(|| format!("could not allocate {n} bytes"))?;
                    let data = buf.data();
                    staged = Some(buf);
                    data
                };
            } else {
                // SAFETY: `headers` has `returned_header_size` entries, 2 * nd of them
                // this output's.
                unsafe {
                    arg.shape = headers.add(at);
                    arg.strides = headers.add(at + nd);
                }
                at += 2 * nd;
            }
            s.staged.push(staged);
            s.args.push(arg);
        }
        let ctx = &mut s.ctx;
        let mark = ctx.scratch_mark();
        let (kernel, config, args) = (self.kernel, self.config, s.args.as_mut_ptr());
        // SAFETY: the kernel contract (`new`): `ctx` is ours for the call; the inputs
        // are valid arrays of their types, borrowed until we return; provided outputs
        // are C-contiguous memory of their static shapes (allocated here, or the
        // caller's) that overlaps no input; returned outputs have no data and storage
        // for their shapes (this function's contract).
        let run = |ctx: &mut PtCtx| unsafe { kernel(ctx, args, n_in as i32, n_out as i32, config) };
        let rc = if self.runs_on_pool(ctx) {
            // (we wait for the pool, and nobody else uses the context and arguments
            // meanwhile)
            let call = CallerWaits((&mut *ctx, run));
            runtime().pool().install(move || {
                let (ctx, run) = call.into_inner();
                run(ctx)
            })
        } else {
            run(ctx)
        };
        ctx.scratch_release(mark);
        let mut result = match rc {
            0 => Ok(()),
            _ => Err(ctx
                .error
                .take()
                .map(|(_, m)| m)
                .unwrap_or_else(|| format!("error {rc}"))),
        };
        // every output: what it is in (also on failure, to release it)
        for (o, t) in self.outputs.iter().enumerate() {
            let raw = s.args[n_in + o];
            let target = s.targets[o];
            let buf = if self.provided[o] {
                s.staged[o].take()
            } else {
                // SAFETY: a returned output's owner is one reference handed to us.
                (!raw.owner.is_null()).then(|| unsafe { OwnedBuf::from_raw(raw.owner) })
            };
            if result.is_err() || target.direct {
                continue;
            }
            result = (|| {
                let buf = buf.ok_or("the function returned no buffer")?;
                let nd = t.ndim();
                // SAFETY: the header's storage: nd lengths and nd strides, which the
                // kernel wrote (returned) or we did (provided).
                let (shape, strides) = unsafe {
                    (
                        std::slice::from_raw_parts(raw.shape, nd).to_vec(),
                        std::slice::from_raw_parts(raw.strides, nd).to_vec(),
                    )
                };
                let array = OwnedArray::in_buffer(buf, raw.data, shape, strides, t.dtype)?;
                if !self.provided[o] {
                    t.check(array.shape())?;
                }
                if target.array.data.is_null() {
                    s.results.push((o, array));
                    return Ok(());
                }
                // SAFETY: the caller's memory, checked as an array to write and
                // borrowed until we return (this function's contract).
                let mut into = unsafe { PtArrayMut::from_raw(&target.array) }?;
                if array.shape() != into.shape() {
                    return Err(format!(
                        "provided output {o}: shape mismatch: the result has shape {:?}, got {:?}",
                        array.shape(),
                        into.shape()
                    ));
                }
                into.copy_from(&array.as_ref())
            })()
            .map_err(|e| format!("output {o}: {e}"));
        }
        if result.is_err() {
            s.results.clear();
        }
        result
    }
}

/// Where calls of functions run: their context (scratch memory, errors, the random
/// state), and what calls keep between them, so that calling a function with static
/// shapes allocates nothing. One call at a time; any function can use it.
#[derive(Default)]
pub struct Session {
    pub ctx: PtCtx,
    /// inputs ++ outputs, as the kernel gets them
    args: Vec<PtArray>,
    /// per output: where the caller wants it
    targets: Vec<Target>,
    /// per output: a buffer we allocated for a provided output whose destination the
    /// kernel can't write directly
    staged: Vec<Option<OwnedBuf>>,
    /// the shapes and strides of the returned outputs (the kernel writes them)
    headers: Vec<i64>,
    /// byte ranges of the inputs and of the memory for outputs of the call
    in_ranges: Vec<(usize, usize)>,
    out_ranges: Vec<(usize, usize)>,
    /// the new arrays of the call (`call` hands them out)
    results: Vec<(usize, OwnedArray)>,
}

// SAFETY: the context is Send; between calls the pointers kept are never used again
// (`call` clears them first), and the rest are buffer references, which are Send.
unsafe impl Send for Session {}

impl Session {
    pub fn new() -> Session {
        Session::default()
    }
}

#[derive(Clone, Copy)]
struct Target {
    /// the caller's memory (an array to write, borrowed for the call); null data: a new
    /// array
    array: PtArray,
    /// the kernel writes the caller's memory itself
    direct: bool,
}
