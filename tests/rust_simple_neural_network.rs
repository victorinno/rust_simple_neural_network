use ndarray::Array1;
use neural_network::neural_network::neural_network::{Layer, Loss, NeuralNetwork, ParamOperation};
use rust_simple_neural_network::neural_network;

#[test]
fn neural_network_creation() {
    let l: Array1<Layer<ParamOperation>> = Array1::from_shape_fn(15, |_i| Layer {
        seed: 0,
        input: Array1::zeros(15),
        output: Array1::zeros(15),
        operations: Array1::from_shape_fn(15, |_i| ParamOperation {
            input: Array1::zeros(15),
            output: Array1::zeros(15),
        }),
    });
    let nn = NeuralNetwork::init(l, Loss {}, 3);
    assert_eq!(nn.layers.len(), 15);
    assert_eq!(nn.layers.get(0).unwrap().seed, 3);
    assert_eq!(nn.seed, 3);
}
