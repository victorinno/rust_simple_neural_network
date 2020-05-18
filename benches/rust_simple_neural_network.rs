extern crate rand;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_simple_neural_network::*;
use ndarray::Array1;
use rand::Rng;

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let n = &Array1::from_shape_fn((15), |(i)| rng.gen::<f64>());
    let m = &Array1::from_shape_fn((15), |(i)| rng.gen::<f64>());
    c.bench_function("mean_absolute_error 15 ", |b| b.iter(|| mean_absolute_error(black_box(n), black_box(m))));

    let n = &Array1::from_shape_fn((150), |(i)| rng.gen::<f64>());
    let m = &Array1::from_shape_fn((150), |(i)| rng.gen::<f64>());
    c.bench_function("mean_absolute_error 150 ", |b| b.iter(|| mean_absolute_error(black_box(n), black_box(m))));

    let n = &Array1::from_shape_fn((1500), |(i)| rng.gen::<f64>());
    let m = &Array1::from_shape_fn((1500), |(i)| rng.gen::<f64>());
    c.bench_function("mean_absolute_error 1500 ", |b| b.iter(|| mean_absolute_error(black_box(n), black_box(m))));

    // c.bench_function("reverse2", |b| b.iter(|| reverse2(black_box("I'm hungry!"))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
