use static_typing_tch::tensor_check;
use tch::{kind::Kind, Device, Tensor};

tensor_check! {
#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/failures/*.rs");
}


fn concat_1(left: Tensor<(B, S1)>, right: Tensor<(B, S2)>) -> Tensor<(B, S1 + S2)> {
    Tensor::cat(&[left, right], 1)
}

fn gelu(x: &Tensor<(BS, H)>) -> Tensor<(BS, H)> {
    let y: Tensor<(BS, H)> = 0.79788456 * x * (1.0 + 0.044715 * x * x);
    x * 0.5 * (1.0 + y.tanh())
}

fn transformer_mlp(
    hidden_states: &Tensor<(BS, H)>,
    dense_h_to_4h: &Tensor<(H, H4)>,
    dense_h_to_4h_bias: &Tensor<(H4, )>,
    dense_4h_to_h: &Tensor<(H4, H)>,
    dense_4h_to_h_bias: &Tensor<(H, )>,
) -> Tensor<(BS, H)> {
    let hidden_states = dense_h_to_4h_bias.addmm(&hidden_states, &dense_h_to_4h);
    let hidden_states = gelu(&hidden_states);
    let hidden_states = dense_4h_to_h_bias.addmm(&hidden_states, &dense_4h_to_h);
    hidden_states
}

fn transformer_mlp2(
    hidden_states: &Tensor<(B, S, H)>,
    dense_h_to_4h: &Tensor<(H, H4)>,
    dense_h_to_4h_bias: &Tensor<(H4, )>,
    dense_4h_to_h: &Tensor<(H4, H)>,
    dense_4h_to_h_bias: &Tensor<(H, )>,
) -> Tensor<(B, S, H)> {
    let size = hidden_states.size();
    let hidden_states = hidden_states.view((-1, size[2]));
    let hidden_states = dense_h_to_4h_bias.addmm(&hidden_states, &dense_h_to_4h);
    let hidden_states = gelu(&hidden_states);
    let hidden_states = dense_4h_to_h_bias.addmm(&hidden_states, &dense_4h_to_h);
    let hidden_states = hidden_states.view((size[0], size[1], size[2]));
    hidden_states
}

#[test]
fn concat_tensors() {
    let a = Tensor::ones(&[3, 5], (Kind::Float, Device::Cpu));
    let b = Tensor::ones(&[3, 2], (Kind::Float, Device::Cpu));
    let _c = concat_1(a, b);
}

#[test]
fn tensor_call() {
    let hidden_states = Tensor::ones(&[3 * 5, 24], (Kind::Float, Device::Cpu));
    let dense_h_to_4h = Tensor::ones(&[24, 4 * 24], (Kind::Float, Device::Cpu));
    let dense_h_to_4h_bias = Tensor::ones(&[4 * 24], (Kind::Float, Device::Cpu));

    let t = Tensor::addmm(&dense_h_to_4h_bias, &hidden_states, &dense_h_to_4h);

    assert_eq!(t.size(), &[15, 96]);

}

#[test]
fn attention() {
    let hidden_states = Tensor::ones(&[3 * 5, 24], (Kind::Float, Device::Cpu));
    let dense_h_to_4h = Tensor::ones(&[24, 4 * 24], (Kind::Float, Device::Cpu));
    let dense_h_to_4h_bias = Tensor::ones(&[4 * 24], (Kind::Float, Device::Cpu));

    let dense_4h_to_h = Tensor::ones(&[4 * 24, 24], (Kind::Float, Device::Cpu));
    let dense_4h_to_h_bias = Tensor::ones(&[24], (Kind::Float, Device::Cpu));

    let _c = transformer_mlp(
        &hidden_states,
        &dense_h_to_4h,
        &dense_h_to_4h_bias,
        &dense_4h_to_h,
        &dense_4h_to_h_bias,
    );

    let hidden_states = Tensor::ones(&[3, 5, 24], (Kind::Float, Device::Cpu));
    let _c = transformer_mlp2(
        &hidden_states,
        &dense_h_to_4h,
        &dense_h_to_4h_bias,
        &dense_4h_to_h,
        &dense_4h_to_h_bias,
    );
}

struct Linear{
    weight: Tensor<(O, I)>,
    bias: Tensor<(O, )>
}

impl Linear{
    fn new(weight: Tensor<(O, I)>, bias: Tensor<(O, )>) -> Self{
        Self{
            weight,
            bias
        }
    }

    fn forward(&self, x: &Tensor<(B, I)>) -> Tensor<(B, O)>{
        x.linear(&self.weight, Some(&self.bias))
    }
}
#[test]
fn user_defined_struct() {
    let weight = Tensor::ones(&[5, 3], (Kind::Float, Device::Cpu));
    let bias = Tensor::ones(&[5], (Kind::Float, Device::Cpu));
    let linear =  Linear::new(weight, bias);

    let x = Tensor::ones(&[2, 3], (Kind::Float, Device::Cpu));
    let out = linear.forward(&x);

    let expected = Tensor::ones(&[2, 5], (Kind::Float, Device::Cpu)) * 4;
    assert_eq!(out, expected);
}

}
