//! Typed indices and index-typed vectors.
//!
//! Every integer that identifies something (a value in a graph, an input of a node, an
//! axis, a slot in the arena) gets its own newtype, so they can't be mixed up.

use std::marker::PhantomData;

/// Define `pub struct $name(pub u32)` usable as an `IndexVec` index.
#[macro_export]
macro_rules! define_index {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        pub struct $name(pub u32);

        impl $crate::v2::index::Idx for $name {
            fn new(i: usize) -> Self {
                $name(u32::try_from(i).expect(concat!(stringify!($name), " overflow")))
            }
            fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

pub trait Idx: Copy + Eq + std::fmt::Debug + 'static {
    fn new(i: usize) -> Self;
    fn index(self) -> usize;
}

/// A `Vec<T>` that can only be indexed with `I`.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct IndexVec<I: Idx, T> {
    raw: Vec<T>,
    _i: PhantomData<fn(&I)>,
}

impl<I: Idx, T> IndexVec<I, T> {
    pub fn new() -> Self {
        IndexVec { raw: Vec::new(), _i: PhantomData }
    }

    pub fn from_vec(raw: Vec<T>) -> Self {
        IndexVec { raw, _i: PhantomData }
    }

    pub fn push(&mut self, v: T) -> I {
        self.raw.push(v);
        I::new(self.raw.len() - 1)
    }

    pub fn len(&self) -> usize {
        self.raw.len()
    }

    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    pub fn get(&self, i: I) -> Option<&T> {
        self.raw.get(i.index())
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.raw.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.raw.iter_mut()
    }

    pub fn iter_enumerated(&self) -> impl Iterator<Item = (I, &T)> + '_ {
        self.raw.iter().enumerate().map(|(i, v)| (I::new(i), v))
    }

    pub fn indices(&self) -> impl Iterator<Item = I> + 'static {
        (0..self.raw.len()).map(I::new)
    }

    pub fn as_slice(&self) -> &[T] {
        &self.raw
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.raw
    }

    pub fn into_vec(self) -> Vec<T> {
        self.raw
    }
}

impl<I: Idx, T> Default for IndexVec<I, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Idx, T> std::ops::Index<I> for IndexVec<I, T> {
    type Output = T;
    fn index(&self, i: I) -> &T {
        &self.raw[i.index()]
    }
}

impl<I: Idx, T> std::ops::IndexMut<I> for IndexVec<I, T> {
    fn index_mut(&mut self, i: I) -> &mut T {
        &mut self.raw[i.index()]
    }
}

impl<I: Idx, T> FromIterator<T> for IndexVec<I, T> {
    fn from_iter<It: IntoIterator<Item = T>>(it: It) -> Self {
        IndexVec::from_vec(it.into_iter().collect())
    }
}

impl<'a, I: Idx, T> IntoIterator for &'a IndexVec<I, T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.raw.iter()
    }
}

impl<I: Idx, T: std::fmt::Debug> std::fmt::Debug for IndexVec<I, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.raw.fmt(f)
    }
}
