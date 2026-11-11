/// Element types of arrays the runtime handles. Names match numpy dtype names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
        matches!(
            self,
            DType::Int64 | DType::Int32 | DType::Int16 | DType::Int8
        )
    }

    pub fn bits(self) -> u32 {
        match self {
            DType::Bool => 1,
            _ => 8 * self.itemsize() as u32,
        }
    }

    /// Stable integer code: `PtArray::dtype`, `PT_FLOAT64` etc. in the C header.
    pub fn code(self) -> i32 {
        self as i32
    }

    /// The dtype with this `code`, if there is one.
    pub fn from_code(code: i32) -> Option<DType> {
        Some(match code {
            0 => DType::Float64,
            1 => DType::Float32,
            2 => DType::Int64,
            3 => DType::Int32,
            4 => DType::Int16,
            5 => DType::Int8,
            6 => DType::Uint8,
            7 => DType::Bool,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_round_trip() {
        for code in 0..8 {
            assert_eq!(DType::from_code(code).unwrap().code(), code);
        }
        assert!(DType::from_code(8).is_none());
        assert!(DType::from_code(-1).is_none());
    }
}
