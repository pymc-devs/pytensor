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
//! new buffers.

pub mod capi;

use crate::abi::{KernelFn, OwnedBuf, PtArray, c_strides, checked_nbytes};
use crate::array::{OwnedArray, PtArrayMut, PtArrayRef};
use crate::dtype::DType;
use crate::runtime::PtCtx;

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
        self.shape.iter().map(|s| s.map_or(-1, |n| n as i64)).collect()
    }

    fn check(&self, shape: &[i64]) -> R<()> {
        if shape.len() != self.ndim() {
            return Err(format!(
                "expected {} dimensions, got {}",
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
    /// reference for the others (the kernel contract).
    pub unsafe fn new(
        address: usize,
        config: usize,
        inputs: Vec<ArgType>,
        contiguous_inputs: Vec<bool>,
        outputs: Vec<ArgType>,
        provided: Vec<bool>,
    ) -> R<Function> {
        // SAFETY: a null or a function pointer of this type (the function's contract).
        let kernel = unsafe { std::mem::transmute::<usize, Option<KernelFn>>(address) }
            .ok_or("null function")?;
        if contiguous_inputs.len() != inputs.len() || provided.len() != outputs.len() {
            return Err("one flag per argument".into());
        }
        if let Some(o) = (0..outputs.len()).find(|&o| provided[o] && outputs[o].static_shape().is_none()) {
            return Err(format!("output {o}: provided, but its shape isn't static"));
        }
        Ok(Function {
            kernel,
            config: config as *const u8,
            inputs,
            contiguous_inputs,
            outputs,
            provided,
        })
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

    /// Run the function. Per output: `Some` new array for `Destination::New`, `None`
    /// when it was written into the memory given.
    pub fn call(
        &self,
        ctx: &mut PtCtx,
        inputs: &[PtArrayRef],
        outputs: Vec<Destination>,
    ) -> R<Vec<Option<OwnedArray>>> {
        if inputs.len() != self.n_inputs() || outputs.len() != self.n_outputs() {
            return Err(format!(
                "expected {} inputs and {} outputs, got {} and {}",
                self.n_inputs(),
                self.n_outputs(),
                inputs.len(),
                outputs.len()
            ));
        }
        for (k, a) in inputs.iter().enumerate() {
            self.check_input(k, a)?;
        }
        // shape ++ strides of every argument
        let mut storage: Vec<Vec<i64>> = Vec::with_capacity(inputs.len() + outputs.len());
        let mut args: Vec<PtArray> = Vec::with_capacity(inputs.len() + outputs.len());
        for a in inputs {
            storage.push([a.shape(), a.strides()].concat());
        }
        // per output: the buffer of provided memory we allocated, and where it goes
        let mut buffers: Vec<Option<OwnedBuf>> = Vec::with_capacity(outputs.len());
        let mut destinations = Vec::with_capacity(outputs.len());
        for (o, dest) in outputs.into_iter().enumerate() {
            let t = &self.outputs[o];
            if let Destination::Into(m) = &dest {
                if m.dtype() != t.dtype {
                    return Err(format!(
                        "provided output {o}: an array of {} is expected, got one of {}",
                        t.dtype.numpy_name(),
                        m.dtype().numpy_name()
                    ));
                }
                t.check(m.shape())
                    .map_err(|e| format!("provided output {o}: {e}"))?;
            }
            if !self.provided[o] {
                storage.push(vec![-1; 2 * t.ndim()]);
                buffers.push(None);
                destinations.push(dest);
                continue;
            }
            let shape = t.static_shape().unwrap_or_default();
            let itemsize = t.dtype.itemsize();
            storage.push([shape.clone(), c_strides(&shape, itemsize as i64)].concat());
            match dest {
                // written directly
                Destination::Into(m) if m.is_c_contiguous() => {
                    buffers.push(None);
                    destinations.push(Destination::Into(m));
                }
                dest => {
                    let n = checked_nbytes(&shape, itemsize)
                        .ok_or_else(|| crate::abi::bad_size("output", &shape))?;
                    let buf = ctx
                        .alloc(n)
                        .ok_or_else(|| format!("could not allocate {n} bytes"))?;
                    buffers.push(Some(buf));
                    destinations.push(dest);
                }
            }
        }
        for (k, a) in inputs.iter().enumerate() {
            let nd = a.ndim();
            let st = storage[k].as_mut_ptr();
            args.push(PtArray {
                // (never written through: the kernel doesn't write its inputs)
                data: a.as_ptr() as *mut u8,
                shape: st,
                // SAFETY: the storage has 2 * ndim entries.
                strides: unsafe { st.add(nd) },
                owner: std::ptr::null_mut(),
                ndim: nd as i32,
                dtype: a.dtype().code(),
            });
        }
        for (o, t) in self.outputs.iter().enumerate() {
            let nd = t.ndim();
            let st = storage[inputs.len() + o].as_mut_ptr();
            let data = match (&buffers[o], &mut destinations[o]) {
                (Some(buf), _) => buf.data(),
                (None, Destination::Into(m)) if self.provided[o] => m.as_mut_ptr(),
                _ => std::ptr::null_mut(),
            };
            args.push(PtArray {
                data,
                shape: st,
                // SAFETY: the storage has 2 * ndim entries.
                strides: unsafe { st.add(nd) },
                owner: std::ptr::null_mut(),
                ndim: nd as i32,
                dtype: t.dtype.code(),
            });
        }
        let mark = ctx.scratch_mark();
        // SAFETY: the kernel contract (`new`): `ctx` is ours for the call; the inputs
        // are valid arrays of their types (checked above), borrowed until we return;
        // provided outputs are C-contiguous memory of their static shapes (allocated
        // here, or the caller's, checked above) that overlaps no input (the callers
        // make sure); returned outputs have no data and storage for their shapes.
        let rc = unsafe {
            (self.kernel)(
                ctx,
                args.as_mut_ptr(),
                inputs.len() as i32,
                self.n_outputs() as i32,
                self.config,
            )
        };
        ctx.scratch_release(mark);
        let n_in = inputs.len();
        // the buffers the kernel handed over (also on failure)
        let mut returned: Vec<Option<OwnedBuf>> = (0..self.n_outputs())
            .map(|o| {
                let owner = args[n_in + o].owner;
                // SAFETY: a returned output's owner is one reference handed to us.
                (!self.provided[o] && !owner.is_null()).then(|| unsafe { OwnedBuf::from_raw(owner) })
            })
            .collect();
        if rc != 0 {
            let msg = ctx
                .error
                .take()
                .map(|(_, m)| m)
                .unwrap_or_else(|| format!("error {rc}"));
            return Err(msg);
        }
        let mut results = Vec::with_capacity(self.n_outputs());
        for (o, dest) in destinations.into_iter().enumerate() {
            let t = &self.outputs[o];
            let raw = &args[n_in + o];
            let nd = t.ndim();
            let shape = storage[n_in + o][..nd].to_vec();
            let strides = storage[n_in + o][nd..].to_vec();
            let array = match (buffers[o].take(), returned[o].take()) {
                (Some(buf), _) => OwnedArray::in_buffer(buf, raw.data, shape, strides, t.dtype),
                (None, Some(buf)) => {
                    t.check(&shape).map_err(|m| format!("output {o}: {m}"))?;
                    OwnedArray::in_buffer(buf, raw.data, shape, strides, t.dtype)
                }
                (None, None) if self.provided[o] => {
                    results.push(None); // written into the caller's memory
                    continue;
                }
                (None, None) => Err("the function returned no buffer".into()),
            }
            .map_err(|e| format!("output {o}: {e}"))?;
            match dest {
                Destination::New => results.push(Some(array)),
                Destination::Into(mut m) => {
                    if array.shape() != m.shape() {
                        return Err(format!(
                            "provided output {o}: shape mismatch: the result has shape {:?}, got {:?}",
                            array.shape(),
                            m.shape()
                        ));
                    }
                    m.copy_from(&array.as_ref())?;
                    results.push(None);
                }
            }
        }
        Ok(results)
    }
}
