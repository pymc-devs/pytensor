use serde::{Deserialize, Serialize};

/// Element types supported by the PoC. Names match numpy dtype names.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    Float64,
    Float32,
    Int64,
    Int32,
    Int16,
    Int8,
    Uint8,
    Bool,
}

impl DType {
    pub fn itemsize(self) -> usize {
        match self {
            DType::Float64 | DType::Int64 => 8,
            DType::Float32 | DType::Int32 => 4,
            DType::Int16 => 2,
            DType::Int8 | DType::Uint8 | DType::Bool => 1,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, DType::Float64 | DType::Float32)
    }

    pub fn is_signed_int(self) -> bool {
        matches!(self, DType::Int64 | DType::Int32 | DType::Int16 | DType::Int8)
    }

    pub fn bits(self) -> u32 {
        match self {
            DType::Bool => 1,
            _ => 8 * self.itemsize() as u32,
        }
    }

    /// Stable integer code used in kernel params.
    pub fn code(self) -> i32 {
        self as i32
    }

    pub fn from_code(c: i32) -> Option<DType> {
        use DType::*;
        [Float64, Float32, Int64, Int32, Int16, Int8, Uint8, Bool].get(c as usize).copied()
    }

    pub fn from_name(s: &str) -> Option<DType> {
        use DType::*;
        Some(match s {
            "float64" => Float64,
            "float32" => Float32,
            "int64" => Int64,
            "int32" => Int32,
            "int16" => Int16,
            "int8" => Int8,
            "uint8" => Uint8,
            "bool" => Bool,
            _ => return None,
        })
    }

    pub fn numpy_name(self) -> &'static str {
        match self {
            DType::Float64 => "float64",
            DType::Float32 => "float32",
            DType::Int64 => "int64",
            DType::Int32 => "int32",
            DType::Int16 => "int16",
            DType::Int8 => "int8",
            DType::Uint8 => "uint8",
            DType::Bool => "bool",
        }
    }

    pub fn from_numpy(kind: u8, itemsize: usize) -> Option<DType> {
        Some(match (kind, itemsize) {
            (b'f', 8) => DType::Float64,
            (b'f', 4) => DType::Float32,
            (b'i', 8) => DType::Int64,
            (b'i', 4) => DType::Int32,
            (b'i', 2) => DType::Int16,
            (b'i', 1) => DType::Int8,
            (b'u', 1) => DType::Uint8,
            (b'b', 1) => DType::Bool,
            _ => return None,
        })
    }

    /// numpy-style promotion of two dtypes (used for comparisons inside composites).
    pub fn upcast(a: DType, b: DType) -> DType {
        use DType::*;
        if a == b {
            return a;
        }
        match (a.is_float(), b.is_float()) {
            (true, true) => {
                if a == Float64 || b == Float64 {
                    Float64
                } else {
                    Float32
                }
            }
            (true, false) | (false, true) => {
                let (f, i) = if a.is_float() { (a, b) } else { (b, a) };
                if f == Float64 || i.bits() >= 32 {
                    Float64
                } else {
                    Float32
                }
            }
            (false, false) => {
                if a == Bool {
                    return b;
                }
                if b == Bool {
                    return a;
                }
                let signed = a.is_signed_int() || b.is_signed_int();
                let mut bits = a.bits().max(b.bits());
                // uint8 mixed with a signed type of the same width needs one more step
                if signed && (a == Uint8 || b == Uint8) && bits == 8 {
                    bits = 16;
                }
                match bits {
                    8 => {
                        if signed {
                            Int8
                        } else {
                            Uint8
                        }
                    }
                    16 => Int16,
                    32 => Int32,
                    _ => Int64,
                }
            }
        }
    }
}
