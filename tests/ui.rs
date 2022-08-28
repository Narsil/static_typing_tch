use static_typing_tch::tensor_check_fn;
use tch::{kind::Kind, Device, Tensor};

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/failures/*.rs");
}

#[tensor_check_fn]
fn concat_1(
    left: Tensor<(B, S1), K, D>,
    right: Tensor<(B, S2), K, D>,
) -> Tensor<(B, S1 + S2), K, D> {
    Tensor::cat(&[left, right], 1)
}

#[test]
fn concat_tensors() {
    let a = Tensor::ones(&[3, 5], (Kind::Float, Device::Cpu));
    let b = Tensor::ones(&[3, 2], (Kind::Float, Device::Cpu));
    let _c = concat_1(a, b);
}
