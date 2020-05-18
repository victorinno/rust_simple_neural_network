pub mod neural_network {
    use ndarray::Array1;
    #[derive(Debug, Clone)]
    pub struct Loss {}
    #[derive(Debug, Clone)]
    pub struct Layer {
        pub seed: u32,
    }
    #[derive(Debug, Clone)]
    pub struct NeuralNetwork {
        pub layers: Array1<Layer>,
        pub loss: Loss,
        pub seed: u32,
    }
    impl NeuralNetwork {
        pub fn init(layers: Array1<Layer>, loss: Loss, seed: u32) -> NeuralNetwork {
            let l: Array1<Layer> = layers.mapv_into(|mut l| {
                l.seed = seed.clone();
                l
            });
            NeuralNetwork {
                layers: l,
                loss: loss,
                seed: seed,
            }
        }
    }
}
