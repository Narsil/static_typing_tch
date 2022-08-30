use static_typing_tch::tensor_check_fn;
use tch::{kind::Kind, Device, Tensor};

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/failures/*.rs");
}

#[tensor_check_fn]
fn concat_1(left: Tensor<(B, S1)>, right: Tensor<(B, S2)>) -> Tensor<(B, S1 + S2)> {
    Tensor::cat(&[left, right], 1)
}

#[tensor_check_fn]
fn gelu(x: &Tensor<(B, S, H)>) -> Tensor<(B, S, H)> {
    let y: Tensor = 0.79788456 * x * (1.0 + 0.044715 * x * x);
    x * 0.5 * (1.0 + y.tanh())
}

#[tensor_check_fn]
fn transformer_mlp(
    hidden_states: &Tensor<(B, S, H)>,
    dense_h_to_4h: &Tensor<(H, H4)>,
    dense_h_to_4h_bias: &Tensor<(H, U1)>,
    dense_4h_to_h: &Tensor<(H3, H)>,
    dense_4h_to_h_bias: &Tensor<(H4, U1)>,
) -> Tensor<(B, S, H)> {
    let hidden_states = dense_h_to_4h_bias.addmm(&hidden_states, &dense_h_to_4h);
    let hidden_states = gelu(&hidden_states);
    let hidden_states = dense_4h_to_h_bias.addmm(&hidden_states, &dense_4h_to_h);
    hidden_states
}

#[test]
fn concat_tensors() {
    let a = Tensor::ones(&[3, 5], (Kind::Float, Device::Cpu));
    let b = Tensor::ones(&[3, 2], (Kind::Float, Device::Cpu));
    let _c = concat_1(a, b);
}

#[test]
fn attention() {
    let hidden_states = Tensor::ones(&[3, 5, 24], (Kind::Float, Device::Cpu));
    let dense_4h_to_h = Tensor::ones(&[24, 4 * 24], (Kind::Float, Device::Cpu));
    let dense_h_to_4h = Tensor::ones(&[4 * 24, 24], (Kind::Float, Device::Cpu));
    let dense_4h_to_h_bias = Tensor::ones(&[4 * 24, 1], (Kind::Float, Device::Cpu));
    let dense_h_to_4h_bias = Tensor::ones(&[4 * 24, 1], (Kind::Float, Device::Cpu));
    let _c = transformer_mlp(
        &hidden_states,
        &dense_h_to_4h,
        &dense_h_to_4h_bias,
        &dense_4h_to_h,
        &dense_4h_to_h_bias,
    );
}
