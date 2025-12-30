#[rustfmt::skip]
use std::{
    intrinsics::transmute_unchecked,
    marker::PhantomData,
    ops,
    ptr,
};

#[derive(Debug, Clone)]
pub struct Tensor<const N: usize, const D: usize, Shape> {
    pub(crate) data: Box<[f64; N]>,
    pub(crate) _shape_marker: PhantomData<Shape>,
}

impl<const N: usize, const D: usize> From<[f64; N]> for Tensor<N, D, [f64; N]> {
    fn from(value: [f64; N]) -> Self {
        Tensor {
            data: Box::new(value),
            _shape_marker: PhantomData,
        }
    }
}

impl<const N: usize, const D: usize, Shape> Tensor<N, D, Shape>
where
    Shape: ops::Index<usize>,
    <Shape as ops::Index<usize>>::Output: Sized + ArraySize,
{
    pub fn new() -> Self {
        Self {
            data: Box::new([0.; N]),
            _shape_marker: PhantomData,
        }
    }

    pub fn reshape<AltShp>(self) -> Tensor<N, D, AltShp>
    where
        Tensor<N, D, AltShp>: Sized,
    {
        assert_eq!(size_of::<AltShp>(), N * size_of::<f64>());
        let Tensor { data, .. } = self;

        Tensor {
            data,
            _shape_marker: PhantomData::<AltShp>,
        }
    }

    pub fn get(
        &self,
        index: usize,
    ) -> Tensor<
        { <<Shape as ops::Index<usize>>::Output as ArraySize>::SIZE },
        { D - 1 },
        <Shape as ops::Index<usize>>::Output,
    >
    where
        <Shape as ops::Index<usize>>::Output: Sized,
    {
        unsafe {
            let t_data = &transmute_unchecked::<&[f64; N], &Shape>(&*self.data)[index];

            let ptr = t_data as *const _ as *const f64;
            let data_arr: [f64; <<Shape as ops::Index<usize>>::Output as ArraySize>::SIZE] =
                ptr::read(ptr as *const _);

            Tensor {
                data: Box::new(data_arr),
                _shape_marker: PhantomData,
            }
        }
    }

    pub fn at(&self, index: [usize; D]) -> &f64
    where
        Shape: GetFromIndex<D>,
    {
        unsafe { transmute_unchecked::<&[f64; N], &Shape>(&*self.data) }.at(index)
    }

    pub fn set(&mut self, index: [usize; D], value: f64)
    where
        Shape: GetFromIndexMut<D>,
    {
        *(unsafe { transmute_unchecked::<&mut [f64; N], &mut Shape>(&mut *self.data) }
            .at_mut(index)) = value;
    }

    pub fn slice<T: Iterator>(_range: T) {
        todo!()
    }
}

pub trait GetFromIndex<const N: usize> {
    fn at(&self, index: [usize; N]) -> &f64;
}

impl GetFromIndex<0> for f64 {
    fn at(&self, _index: [usize; 0]) -> &f64 {
        self
    }
}

// Recursive case: nested arrays
impl<T, const M: usize, const N: usize> GetFromIndex<N> for [T; M]
where
    T: GetFromIndex<{ N - 1 }>,
{
    default fn at(&self, index: [usize; N]) -> &f64 {
        self[index[0]].at(core::array::from_fn(|i| index[i + 1]))
    }
}

pub trait GetFromIndexMut<const N: usize> {
    fn at_mut(&mut self, index: [usize; N]) -> &mut f64;
}

impl GetFromIndexMut<0> for f64 {
    fn at_mut(&mut self, _index: [usize; 0]) -> &mut f64 {
        self
    }
}

// Recursive case: nested arrays
impl<T, const M: usize, const N: usize> GetFromIndexMut<N> for [T; M]
where
    T: GetFromIndexMut<{ N - 1 }>,
{
    default fn at_mut(&mut self, index: [usize; N]) -> &mut f64 {
        self[index[0]].at_mut(core::array::from_fn(|i| index[i + 1]))
    }
}

pub trait ArraySize {
    const SIZE: usize;
}

// Base case: f64 has "size" 1
impl ArraySize for f64 {
    const SIZE: usize = 1;
}

// Recursive case: [T; N] has size N * T::SIZE
impl<T: ArraySize, const N: usize> ArraySize for [T; N] {
    const SIZE: usize = N * T::SIZE;
}

impl<const N: usize, const D: usize, Shape> Default for Tensor<N, D, Shape>
where
    Shape: ops::Index<usize>,
    <Shape as ops::Index<usize>>::Output: Sized + ArraySize,
{
    fn default() -> Self {
        Self::new()
    }
}

// currently disallow adding two `&Tensor` because I need to overwrite one of them and also be able to own the `data` as I am dealing with `Box<[_; _]>`
impl<const N: usize, const D: usize, Shape> ops::Add<&Tensor<N, D, Shape>> for Tensor<N, D, Shape> {
    type Output = Tensor<N, D, Shape>;
    fn add(mut self, rhs: &Tensor<N, D, Shape>) -> Self::Output {
        for (i, v) in self.data.iter_mut().enumerate() {
            *v += rhs.data[i];
        }

        Self::Output {
            data: self.data,
            _shape_marker: PhantomData,
        }
    }
}

impl<const N: usize, const D: usize, Shape> ops::Div<f64> for Tensor<N, D, Shape> {
    type Output = Tensor<N, D, Shape>;
    fn div(mut self, rhs: f64) -> Self::Output {
        for v in self.data.iter_mut() {
            *v /= rhs;
        }

        Self::Output {
            data: self.data,
            _shape_marker: PhantomData,
        }
    }
}

#[macro_export]
macro_rules! shape_ty {
    ($d:expr) => {
        [f64; $d]
    };

    ($first:expr, $($rest:expr),+ $(,)?) => {
        [$crate::shape_ty!($($rest),+); $first]
    };
}

// don't use this to calculate dims outside of anything. it can often lead to a "cycle detected when computing revealed normalized predicates" error
#[macro_export]
macro_rules! __dim_ty {
    () => { 0 };
    ($head:tt $($tail:tt)*) => { 1 + $crate::__dim_ty!($($tail)*) };
}

#[macro_export]
macro_rules! tensor {
    ($first:expr $(, $rest:expr)* $(,)?) => {
        {
            // number of elements
            const N: usize = $first $( * $rest )*;

            // dimension
            const D: usize = $crate::__dim_ty!($first $($rest )*);

            type Shape = $crate::shape_ty!($first $(, $rest)*);

            <$crate::Tensor::<N, D, Shape>>::new()
        }
    };
}
