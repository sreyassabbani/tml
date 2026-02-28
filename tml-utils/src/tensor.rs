#[rustfmt::skip]
use std::{
    marker::PhantomData,
    mem::size_of,
    ops,
};

use crate::{Assert, Float, IsTrue};

mod sealed {
    use super::Float;

    pub trait Sealed {}

    impl Sealed for Float {}
    impl<T: Sealed, const N: usize> Sealed for [T; N] {}
}

#[derive(Debug)]
pub struct Tensor<const N: usize, const D: usize, Shape> {
    pub(crate) data: Box<[Float; N]>,
    pub(crate) _shape_marker: PhantomData<Shape>,
}

#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a, const N: usize, const D: usize, Shape> {
    data: &'a [Float],
    _shape_marker: PhantomData<Shape>,
}

#[derive(Debug)]
pub struct TensorViewMut<'a, const N: usize, const D: usize, Shape> {
    data: &'a mut [Float],
    _shape_marker: PhantomData<Shape>,
}

impl<const N: usize, const D: usize, Shape> Clone for Tensor<N, D, Shape> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            _shape_marker: PhantomData,
        }
    }
}

impl<const N: usize, const D: usize> From<[Float; N]> for Tensor<N, D, [Float; N]> {
    fn from(value: [Float; N]) -> Self {
        Tensor {
            data: Box::new(value),
            _shape_marker: PhantomData,
        }
    }
}

impl<const N: usize, const D: usize, Shape> Tensor<N, D, Shape>
where
    Shape: ArraySize,
{
    pub fn as_slice(&self) -> &[Float] {
        &self.data[..]
    }

    pub fn as_mut_slice(&mut self) -> &mut [Float] {
        &mut self.data[..]
    }
}

impl<const N: usize, const D: usize, Shape> Tensor<N, D, Shape>
where
    Shape: ArraySize,
    Assert<{ Shape::SIZE == N }>: IsTrue,
{
    pub fn new() -> Self {
        Self {
            data: Box::new([0.; N]),
            _shape_marker: PhantomData,
        }
    }

    pub fn reshape<AltShp>(self) -> Tensor<N, D, AltShp>
    where
        AltShp: ArraySize,
        Tensor<N, D, AltShp>: Sized,
        Assert<{ AltShp::SIZE == N }>: IsTrue,
    {
        debug_assert_eq!(size_of::<AltShp>(), N * size_of::<Float>());
        let Tensor { data, .. } = self;

        Tensor {
            data,
            _shape_marker: PhantomData::<AltShp>,
        }
    }
}

impl<const N: usize, const D: usize, Shape> Tensor<N, D, Shape>
where
    Shape: ArraySize + ShapeIndex<D>,
    Assert<{ Shape::SIZE == N }>: IsTrue,
{
    pub fn view(&self) -> TensorView<'_, N, D, Shape> {
        TensorView::new(self.as_slice())
    }

    pub fn view_mut(&mut self) -> TensorViewMut<'_, N, D, Shape> {
        TensorViewMut::new(self.as_mut_slice())
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
        Shape: ops::Index<usize>,
        <Shape as ops::Index<usize>>::Output: Sized + ArraySize + ShapeIndex<{ D - 1 }>,
    {
        let view = self.get_view(index);
        let mut data_arr =
            [0.0 as Float; <<Shape as ops::Index<usize>>::Output as ArraySize>::SIZE];
        data_arr.copy_from_slice(view.as_slice());
        Tensor {
            data: Box::new(data_arr),
            _shape_marker: PhantomData,
        }
    }

    pub fn get_view(
        &self,
        index: usize,
    ) -> TensorView<
        '_,
        { <<Shape as ops::Index<usize>>::Output as ArraySize>::SIZE },
        { D - 1 },
        <Shape as ops::Index<usize>>::Output,
    >
    where
        Shape: ops::Index<usize>,
        <Shape as ops::Index<usize>>::Output: Sized + ArraySize + ShapeIndex<{ D - 1 }>,
    {
        let sub_size = <<Shape as ops::Index<usize>>::Output as ArraySize>::SIZE;
        let base = index * sub_size;
        assert!(base + sub_size <= N, "index out of bounds");
        TensorView::<
            { <<Shape as ops::Index<usize>>::Output as ArraySize>::SIZE },
            { D - 1 },
            <Shape as ops::Index<usize>>::Output,
        >::new(&self.data[base..base + sub_size])
    }

    pub fn get_view_mut(
        &mut self,
        index: usize,
    ) -> TensorViewMut<
        '_,
        { <<Shape as ops::Index<usize>>::Output as ArraySize>::SIZE },
        { D - 1 },
        <Shape as ops::Index<usize>>::Output,
    >
    where
        Shape: ops::Index<usize>,
        <Shape as ops::Index<usize>>::Output: Sized + ArraySize + ShapeIndex<{ D - 1 }>,
    {
        let sub_size = <<Shape as ops::Index<usize>>::Output as ArraySize>::SIZE;
        let base = index * sub_size;
        assert!(base + sub_size <= N, "index out of bounds");
        TensorViewMut::<
            { <<Shape as ops::Index<usize>>::Output as ArraySize>::SIZE },
            { D - 1 },
            <Shape as ops::Index<usize>>::Output,
        >::new(&mut self.data[base..base + sub_size])
    }

    pub fn at(&self, index: [usize; D]) -> &Float {
        let offset = Shape::offset(index);
        &self.data[offset]
    }

    pub fn set(&mut self, index: [usize; D], value: Float) {
        let offset = Shape::offset(index);
        self.data[offset] = value;
    }

    #[deprecated(note = "Tensor::slice is not implemented yet")]
    pub fn slice<T: Iterator>(_range: T) {
        // Keep this as a non-panicking placeholder while the slicing API is designed.
    }
}

impl<'a, const N: usize, const D: usize, Shape> TensorView<'a, N, D, Shape>
where
    Shape: ArraySize,
{
    fn new(data: &'a [Float]) -> Self {
        debug_assert_eq!(data.len(), N);
        Self {
            data,
            _shape_marker: PhantomData,
        }
    }

    pub fn as_slice(&self) -> &[Float] {
        self.data
    }
}

impl<'a, const N: usize, const D: usize, Shape> TensorView<'a, N, D, Shape>
where
    Shape: ArraySize + ShapeIndex<D>,
{
    pub fn at(&self, index: [usize; D]) -> &Float {
        let offset = Shape::offset(index);
        &self.data[offset]
    }
}

impl<'a, const N: usize, const D: usize, Shape> TensorViewMut<'a, N, D, Shape>
where
    Shape: ArraySize,
{
    fn new(data: &'a mut [Float]) -> Self {
        debug_assert_eq!(data.len(), N);
        Self {
            data,
            _shape_marker: PhantomData,
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [Float] {
        self.data
    }
}

impl<'a, const N: usize, const D: usize, Shape> TensorViewMut<'a, N, D, Shape>
where
    Shape: ArraySize + ShapeIndex<D>,
{
    pub fn at(&self, index: [usize; D]) -> &Float {
        let offset = Shape::offset(index);
        &self.data[offset]
    }

    pub fn set(&mut self, index: [usize; D], value: Float) {
        let offset = Shape::offset(index);
        self.data[offset] = value;
    }
}

pub trait ShapeIndex<const D: usize>: ArraySize + sealed::Sealed {
    fn offset(index: [usize; D]) -> usize;
}

impl ShapeIndex<0> for Float {
    fn offset(_index: [usize; 0]) -> usize {
        0
    }
}

// Recursive case: nested arrays
impl<T, const M: usize, const D: usize> ShapeIndex<D> for [T; M]
where
    T: ShapeIndex<{ D - 1 }> + ArraySize,
{
    fn offset(index: [usize; D]) -> usize {
        let head = index[0];
        assert!(head < M, "index out of bounds");
        head * T::SIZE + T::offset(core::array::from_fn(|i| index[i + 1]))
    }
}

pub trait ArraySize: sealed::Sealed {
    const SIZE: usize;
}

// Base case: Float has "size" 1
impl ArraySize for Float {
    const SIZE: usize = 1;
}

// Recursive case: [T; N] has size N * T::SIZE
impl<T: ArraySize, const N: usize> ArraySize for [T; N] {
    const SIZE: usize = N * T::SIZE;
}

impl<const N: usize, const D: usize, Shape> Default for Tensor<N, D, Shape>
where
    Shape: ArraySize + ShapeIndex<D>,
    Assert<{ Shape::SIZE == N }>: IsTrue,
{
    fn default() -> Self {
        Self::new()
    }
}

// currently disallow adding two `&Tensor` because I need to overwrite one of them and also be able to own the `data` as I am dealing with `Box<[_; _]>`
impl<const N: usize, const D: usize, Shape> ops::Add<&Tensor<N, D, Shape>> for Tensor<N, D, Shape>
where
    Shape: ArraySize,
    Assert<{ Shape::SIZE == N }>: IsTrue,
{
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

impl<const N: usize, const D: usize, Shape> ops::Add<&Tensor<N, D, Shape>> for &Tensor<N, D, Shape>
where
    Shape: ArraySize,
    Assert<{ Shape::SIZE == N }>: IsTrue,
{
    type Output = Tensor<N, D, Shape>;
    fn add(self, rhs: &Tensor<N, D, Shape>) -> Self::Output {
        let mut out = self.clone();
        out += rhs;
        out
    }
}

impl<const N: usize, const D: usize, Shape> ops::AddAssign<&Tensor<N, D, Shape>>
    for Tensor<N, D, Shape>
where
    Shape: ArraySize,
    Assert<{ Shape::SIZE == N }>: IsTrue,
{
    fn add_assign(&mut self, rhs: &Tensor<N, D, Shape>) {
        for (i, v) in self.data.iter_mut().enumerate() {
            *v += rhs.data[i];
        }
    }
}

impl<const N: usize, const D: usize, Shape> ops::Div<Float> for Tensor<N, D, Shape>
where
    Shape: ArraySize,
    Assert<{ Shape::SIZE == N }>: IsTrue,
{
    type Output = Tensor<N, D, Shape>;
    fn div(mut self, rhs: Float) -> Self::Output {
        for v in self.data.iter_mut() {
            *v /= rhs;
        }

        Self::Output {
            data: self.data,
            _shape_marker: PhantomData,
        }
    }
}

impl<const N: usize, const D: usize, Shape> ops::DivAssign<Float> for Tensor<N, D, Shape>
where
    Shape: ArraySize,
    Assert<{ Shape::SIZE == N }>: IsTrue,
{
    fn div_assign(&mut self, rhs: Float) {
        for v in self.data.iter_mut() {
            *v /= rhs;
        }
    }
}

impl<const N: usize, const D: usize, Shape> ops::Mul<&Tensor<N, D, Shape>> for Tensor<N, D, Shape>
where
    Shape: ArraySize,
    Assert<{ Shape::SIZE == N }>: IsTrue,
{
    type Output = Tensor<N, D, Shape>;
    fn mul(mut self, rhs: &Tensor<N, D, Shape>) -> Self::Output {
        for (i, v) in self.data.iter_mut().enumerate() {
            *v *= rhs.data[i];
        }

        Self::Output {
            data: self.data,
            _shape_marker: PhantomData,
        }
    }
}

impl<const N: usize, const D: usize, Shape> ops::Mul<&Tensor<N, D, Shape>> for &Tensor<N, D, Shape>
where
    Shape: ArraySize,
    Assert<{ Shape::SIZE == N }>: IsTrue,
{
    type Output = Tensor<N, D, Shape>;
    fn mul(self, rhs: &Tensor<N, D, Shape>) -> Self::Output {
        let mut out = self.clone();
        out *= rhs;
        out
    }
}

impl<const N: usize, const D: usize, Shape> ops::MulAssign<&Tensor<N, D, Shape>>
    for Tensor<N, D, Shape>
where
    Shape: ArraySize,
    Assert<{ Shape::SIZE == N }>: IsTrue,
{
    fn mul_assign(&mut self, rhs: &Tensor<N, D, Shape>) {
        for (i, v) in self.data.iter_mut().enumerate() {
            *v *= rhs.data[i];
        }
    }
}

impl<const N: usize, const D: usize, Shape> ops::Mul<Float> for Tensor<N, D, Shape>
where
    Shape: ArraySize,
    Assert<{ Shape::SIZE == N }>: IsTrue,
{
    type Output = Tensor<N, D, Shape>;
    fn mul(mut self, rhs: Float) -> Self::Output {
        for v in self.data.iter_mut() {
            *v *= rhs;
        }

        Self::Output {
            data: self.data,
            _shape_marker: PhantomData,
        }
    }
}

impl<const N: usize, const D: usize, Shape> ops::MulAssign<Float> for Tensor<N, D, Shape>
where
    Shape: ArraySize,
    Assert<{ Shape::SIZE == N }>: IsTrue,
{
    fn mul_assign(&mut self, rhs: Float) {
        for v in self.data.iter_mut() {
            *v *= rhs;
        }
    }
}

impl<const N: usize, const D: usize, Shape> Tensor<N, D, Shape>
where
    Shape: ArraySize,
    Assert<{ Shape::SIZE == N }>: IsTrue,
{
    pub fn map_inplace<F>(&mut self, mut f: F)
    where
        F: FnMut(Float) -> Float,
    {
        for v in self.data.iter_mut() {
            *v = f(*v);
        }
    }

    pub fn map<F>(&self, f: F) -> Self
    where
        F: FnMut(Float) -> Float,
    {
        let mut out = self.clone();
        out.map_inplace(f);
        out
    }
}

#[macro_export]
macro_rules! shape_ty {
    ($d:expr) => {
        [$crate::Float; $d]
    };

    ($first:expr, $($rest:expr),+ $(,)?) => {
        [$crate::shape_ty!($($rest),+); $first]
    };
}

#[macro_export]
macro_rules! __count_exprs {
    () => { 0 };
    ($_head:expr $(, $tail:expr)*) => { 1 + $crate::__count_exprs!($($tail),*) };
}

#[macro_export]
macro_rules! tensor {
    ($first:expr $(, $rest:expr)* $(,)?) => {
        {
            // number of elements
            const N: usize = $first $( * $rest )*;

            // dimension
            const D: usize = $crate::__count_exprs!($first $(, $rest )*);

            type Shape = $crate::shape_ty!($first $(, $rest)*);

            <$crate::Tensor::<N, D, Shape>>::new()
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    type T3 = Tensor<24, 3, shape_ty!(2, 3, 4)>;

    #[test]
    fn indexing_views_and_owned_get_match_layout() {
        let mut t = T3::new();
        let mut value = 0.0;
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    t.set([i, j, k], value);
                    value += 1.0;
                }
            }
        }

        assert_eq!(*t.at([1, 2, 3]), 23.0);

        let row = t.get_view(1);
        assert_eq!(*row.at([2, 3]), 23.0);

        let owned = t.get(1);
        assert_eq!(*owned.at([2, 3]), 23.0);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn get_view_panics_on_oob_index() {
        let t = T3::new();
        let _ = t.get_view(2);
    }
}
