pub mod neural_network {
    use ndarray::Array1;
    #[derive(Debug, Clone)]
    pub struct Loss {}
    pub trait Operation {
        fn set_input(&mut self, input: &Array1<f64>);
        fn get_output(&self) -> Array1<f64>;
        fn forward(&mut self, input: &Array1<f64>) -> Array1<f64>;
    }
    #[derive(Debug, Clone)]
    pub struct ParamOperation {
        pub input: Array1<f64>,
        pub output: Array1<f64>,
    }

    impl Operation for ParamOperation {
        fn set_input(&mut self, input: &Array1<f64>) {
            todo!()
        }
        fn get_output(&self) -> Array1<f64> {
            todo!()
        }
        fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
            todo!()
        }
    }

    #[derive(Debug, Clone)]
    pub struct Layer<T>
    where
        T: Operation,
    {
        pub seed: u32,
        pub input: Array1<f64>,
        pub output: Array1<f64>,
        pub operations: Array1<T>,
    }

    impl<T> Layer<T>
    where
        T: Operation + std::clone::Clone,
    {
        pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
            self.input = input.clone();
            let mut result = input.clone();
            self.operations.mapv_inplace(|mut o| {
                result = o.forward(&result);
                o
            });
            self.output = result;
            return self.output.clone();
        }
    }
    #[derive(Debug, Clone)]
    pub struct NeuralNetwork<T>
    where
        T: Operation,
    {
        pub layers: Array1<Layer<T>>,
        pub loss: Loss,
        pub seed: u32,
    }
    impl<T> NeuralNetwork<T>
    where
        T: Operation + std::clone::Clone,
    {
        pub fn init(layers: Array1<Layer<T>>, loss: Loss, seed: u32) -> NeuralNetwork<T> {
            let l: Array1<Layer<T>> = layers.mapv_into(|mut l| {
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
