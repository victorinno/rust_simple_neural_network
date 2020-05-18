use ndarray::Array1;
use rust_simple_neural_network::neural_network;
use neural_network::neural_network::neural_network::{NeuralNetwork, Layer, Loss};


#[test]
fn neural_network_creation() {
    let l: Array1<Layer> =  Array1::from_shape_fn(15, |_i| Layer{ seed: 0}); 
    let nn = NeuralNetwork::init(l, Loss{},3);
    assert_eq!(nn.layers.len(), 15);
    assert_eq!(nn.layers.get(0).unwrap().seed, 3);
    assert_eq!(nn.seed, 3);

}
