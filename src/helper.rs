use burn::tensor::{Int, Tensor, activation::relu, cast::ToElement, kind::Numeric};

pub fn tensor_max_scalar<const D: usize>(x: Tensor<D>, max: f64) -> Tensor<D> {
    relu(x.sub_scalar(max)).add_scalar(max)
}

pub fn tensor_min_scalar<const D: usize>(x: Tensor<D>, min: f64) -> Tensor<D> {
    -tensor_max_scalar(-x, -min)
}

pub fn tensor_max<const D: usize>(x: Tensor<D>, max: Tensor<D>) -> Tensor<D> {
    relu(x - max.clone()) + max
}

pub fn tensor_min<const D: usize>(x: Tensor<D>, min: Tensor<D>) -> Tensor<D> {
    -tensor_max(-x, -min)
}

pub fn tensor_log10<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let ln10 = (10.0f64).ln();
    x.log() / ln10
}

pub fn all_zeros<const D: usize>(x: Tensor<D>) -> bool {
    x.abs().max().into_scalar::<f64>().to_f64() == 0.0
}

pub fn _10pow<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let log10 = (10.0f64).ln();
    (x * log10).exp()
}

pub fn reverse<const D: usize, K: Numeric>(x: Tensor<D, K>, dim: usize) -> Tensor<D, K> {
    let len = x.dims()[dim];
    let indices = -Tensor::<1, Int>::arange(0..len as i64, &x.device()) + (len - 1) as i64;
    x.select(dim, indices)
}
