use static_typing_tch::tensor_check_fn;
use tch::Tensor;

#[tensor_check_fn]
fn transformer_mlp(
    hidden_states: &Tensor<(B, S, H)>,
    dense: &Tensor<(H, H4)>,
    dense_bias: &Tensor<(U1, H4)>,
) -> Tensor<(B, S, H)> {
    let size = hidden_states.size();
    let hidden_states = hidden_states.view((-1, size[1])); // Error here
    let hidden_states = dense_bias.addmm(&hidden_states, &dense);
    let hidden_states = hidden_states.view((size[0], size[1], size[2]));
    hidden_states
}

#[tensor_check_fn]
fn transformer_mlp2(
    hidden_states: &Tensor<(B, S, H)>,
    dense: &Tensor<(H, H4)>,
    dense_bias: &Tensor<(U1, H4)>,
) -> Tensor<(B, S, H)> {
    let size = hidden_states.size();
    let hidden_states = hidden_states.view((size[2], -1)); // Error here
    let hidden_states = dense_bias.addmm(&hidden_states, &dense);
    let hidden_states = hidden_states.view((size[0], size[1], size[2]));
    hidden_states
}

#[tensor_check_fn]
fn transformer_mlp3(
    hidden_states: &Tensor<(B, S, H)>,
    dense: &Tensor<(H, H4)>,
    dense_bias: &Tensor<(U1, H4)>,
) -> Tensor<(B, S, H)> {
    let size = hidden_states.size();
    let hidden_states = hidden_states.view((-1, size[2]));
    let hidden_states = dense_bias.addmm(&hidden_states, &dense);
    let hidden_states = hidden_states.view((size[0], size[1])); // Error here missing 1 dim
    hidden_states
}

fn main() {}
