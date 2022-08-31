use static_typing_tch::tensor_check_fn;
use tch::Tensor;

#[tensor_check_fn]
fn gelu(x: &Tensor<(B, S, H)>) -> Tensor<(B, S, H)> {
    let y: Tensor = 0.79788456 * x * (1.0 + 0.044715 * x * x);
    x * 0.5 * (1.0 + y.tanh())
}

#[tensor_check_fn]
fn transformer_mlp(
    hidden_states: &Tensor<(BS, H)>,
    dense_h_to_4h: &Tensor<(H, H4)>,
    dense_h_to_4h_bias: &Tensor<(U1, H4)>,
    dense_4h_to_h: &Tensor<(H3, H)>, // Dimension mismatch it should be h3
    dense_4h_to_h_bias: &Tensor<(U1, H)>,
) -> Tensor<(B, S, H)> {
    let hidden_states = dense_h_to_4h_bias.addmm(&hidden_states, &dense_h_to_4h);
    let hidden_states = gelu(&hidden_states);
    let hidden_states = dense_4h_to_h_bias.addmm(&hidden_states, &dense_4h_to_h);
    hidden_states
}

#[tensor_check_fn]
fn transformer_mlp2(
    hidden_states: &Tensor<(BS, H)>,
    dense_h_to_4h: &Tensor<(H, H4)>,
    dense_h_to_4h_bias: &Tensor<(U1, H4)>,
    dense_4h_to_h: &Tensor<(H4, H)>,
    dense_4h_to_h_bias: &Tensor<(U1, H)>,
) -> Tensor<(B, S, H)> {
    let hidden_states = dense_h_to_4h_bias.addmm(&hidden_states, &dense_h_to_4h);
    let hidden_states = gelu(&hidden_states);
    let hidden_states = dense_4h_to_h_bias.addmm(&hidden_states, &dense_h_to_4h); // Dimension mismatch wrong argument

    hidden_states
}

#[tensor_check_fn]
fn transformer_mlp3(
    hidden_states: &Tensor<(B, S, H)>,
    dense_h_to_4h: &Tensor<(H, H4)>,
    dense_h_to_4h_bias: &Tensor<(U1, H4)>,
    dense_4h_to_h: &Tensor<(H4, H)>,
    dense_4h_to_h_bias: &Tensor<(U1, H)>,
) -> Tensor<(B, S, H)> {
    let hidden_states = dense_h_to_4h_bias.addmm(&hidden_states, &dense_h_to_4h); //Mismatch addmm
                                                                                  //supports only
                                                                                  //matrices, a
                                                                                  //`.view(-1, ..)`
                                                                                  //is missing
    let hidden_states = gelu(&hidden_states);
    let hidden_states = dense_4h_to_h_bias.addmm(&hidden_states, &dense_h_to_4h);

    hidden_states
}

fn main() {}
