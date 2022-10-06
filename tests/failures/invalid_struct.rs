use static_typing_tch::tensor_check;
use tch::{kind::Kind, Device, Tensor};

tensor_check! {
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

fn user_defined_struct() {
    let weight = Tensor::ones(&[5, 3], (Kind::Float, Device::Cpu));
    let bias = Tensor::ones(&[5], (Kind::Float, Device::Cpu));
    let linear =  Linear::new(weight, bias);

    let x = Tensor::ones(&[4, 3], (Kind::Float, Device::Cpu));
    let out = linear.forward(&x);

    let x = Tensor::ones(&[2, 4], (Kind::Float, Device::Cpu));
    let out = linear.forward(&x);

    let expected = Tensor::ones(&[2, 5], (Kind::Float, Device::Cpu)) * 4;
    assert_eq!(out, expected);
}
fn main() {}
}
