use static_typing_tch::tensor_check_fn;
use tch::{kind::Kind, Device, Tensor};

#[tensor_check_fn(a, b)]
fn add() {
    // The dimensions do not match
    let a = Tensor::ones(&[2, 3], (Kind::Float, Device::Cpu));
    let b = Tensor::ones(&[3, 2], (Kind::Float, Device::Cpu));

    let _c = a + b;
}

fn main() {
    add();
}
