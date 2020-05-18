pub mod neural_network {
    use ndarray::Array1;
    #[derive(Debug, Clone)]
    pub struct Loss {}
    #[derive(Debug, Clone)]
    pub struct Operation {
        input: Array1<f64>,
        output: Array1<f64>,
    }

    impl Operation {
        pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
            self.input = input.clone();
            
            self.calculate_output();

            return self.output.clone();
        }

        fn calculate_output(&mut self) {
            unimplemented!();
        }
    }

    #[derive(Debug, Clone)]
    pub struct Layer {
        pub seed: u32,
        input: Array1<f64>,
        output: Array1<f64>,
        operations: Array1<Operation>
    }

    impl Layer {
        pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
            self.input = input.clone();
            let mut result = input.clone();
            self.operations.mapv_inplace(|o| {
                result = o.forward(&result);
                o
            });
            self.output = result;
            return self.output.clone();
        }
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

        pub fn fowrad(&mut self, x_batch: &Array1<f64>) -> Array1<f64> {
            let mut out: Array1<f64> = x_batch.clone();
            self.layers.mapv_inplace(|mut l| {
                out = l.forward(&out);
                l
            });
            out
        }
    }
}
