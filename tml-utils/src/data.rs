use crate::Float;

#[derive(Debug, Clone)]
pub struct Sample<const IN: usize, const OUT: usize> {
    pub input: [Float; IN],
    pub target: [Float; OUT],
}

impl<const IN: usize, const OUT: usize> Sample<IN, OUT> {
    pub fn new(input: [Float; IN], target: [Float; OUT]) -> Self {
        Self { input, target }
    }
}

impl<const IN: usize, const OUT: usize> From<(&[Float; IN], &[Float; OUT])> for Sample<IN, OUT> {
    fn from(value: (&[Float; IN], &[Float; OUT])) -> Self {
        Self::new(*value.0, *value.1)
    }
}

impl<const IN: usize, const OUT: usize> From<([Float; IN], [Float; OUT])> for Sample<IN, OUT> {
    fn from(value: ([Float; IN], [Float; OUT])) -> Self {
        Self::new(value.0, value.1)
    }
}
