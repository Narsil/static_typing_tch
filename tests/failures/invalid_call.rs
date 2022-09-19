use static_typing_tch::tensor_check;
use tch::{kind::Kind, Device, Tensor};

tensor_check! {
fn gelu(x: &Tensor<(B, S, H)>) -> Tensor<(B, S, H)> {
    let y: Tensor<(B, S, H)> = 0.79788456 * x * (1.0 + 0.044715 * x * x);
    x * 0.5 * (1.0 + y.tanh())
}


fn main(){
    let hidden_states = Tensor::ones(&[3, 24], (Kind::Float, Device::Cpu));
    let _hidden_states = gelu(&hidden_states);
}
}
