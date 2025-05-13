use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use rand::rng;
use rust_tensor_lib::{backend::cpu::CpuBackend, ops, CpuTensor, Tensor};
#[cfg(feature = "cuda")]
use rust_tensor_lib::{
    backend::cuda::{init_context, CudaBackend, CudaContextGuard},
    CudaTensor,
};

// Helper function to create random CPU tensor
fn create_random_cpu(shape: &[usize]) -> CpuTensor {
    let size = shape.iter().product();
    let mut rng_instance = rng();
    let data: Vec<f32> = (0..size).map(|_| rng_instance.random::<f32>()).collect();
    Tensor::<CpuBackend>::from_vec(data, shape, false).unwrap()
}

// Helper function to create random CUDA tensor
#[cfg(feature = "cuda")]
fn create_random_cuda(shape: &[usize]) -> CudaTensor {
    let size = shape.iter().product();
    let mut rng_instance = rng();
    let data: Vec<f32> = (0..size).map(|_| rng_instance.random::<f32>()).collect();
    Tensor::<CudaBackend>::from_vec(data, shape, false).unwrap()
}

fn bench_matrix_multiply(c: &mut Criterion) {
    let shapes = [
        ([256, 256], "256"),
        ([1024, 1024], "1024"),
        ([4096, 4096], "4096"),
    ];

    let mut group = c.benchmark_group("matrix_multiply");

    for (shape, size) in shapes.iter() {
        // CPU benchmarks
        {
            let a = create_random_cpu(shape);
            let b = create_random_cpu(shape);

            group.bench_function(format!("cpu_matmul_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::matmul(black_box(&a), black_box(&b))).unwrap();
                });
            });
        }

        // GPU benchmarks
        #[cfg(feature = "cuda")]
        {
            init_context(0).unwrap();
            let _guard = CudaContextGuard::new().unwrap();

            let a = create_random_cuda(shape);
            let b = create_random_cuda(shape);

            group.bench_function(&format!("gpu_matmul_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::matmul(black_box(&a), black_box(&b))).unwrap();
                });
            });
        }
    }
    group.finish();
}

fn bench_element_wise(c: &mut Criterion) {
    let shapes = [
        ([256, 256], "256"),
        ([1024, 1024], "1024"),
        ([4096, 4096], "4096"),
    ];

    let mut group = c.benchmark_group("element_wise");

    for (shape, size) in shapes.iter() {
        // CPU benchmarks
        {
            let a = create_random_cpu(shape);
            let b = create_random_cpu(shape);

            group.bench_function(format!("cpu_add_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::add(black_box(&a), black_box(&b))).unwrap();
                });
            });

            group.bench_function(format!("cpu_mul_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::mul(black_box(&a), black_box(&b))).unwrap();
                });
            });

            group.bench_function(format!("cpu_div_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::div(black_box(&a), black_box(&b))).unwrap();
                });
            });

            group.bench_function(format!("cpu_sub_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::sub(black_box(&a), black_box(&b))).unwrap();
                });
            });

            group.bench_function(format!("cpu_relu_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::relu(black_box(&a))).unwrap();
                });
            });
        }

        // GPU benchmarks
        #[cfg(feature = "cuda")]
        {
            init_context(0).unwrap();
            let _guard = CudaContextGuard::new().unwrap();

            let a = create_random_cuda(shape);
            let b = create_random_cuda(shape);

            group.bench_function(&format!("gpu_add_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::add(black_box(&a), black_box(&b))).unwrap();
                });
            });

            group.bench_function(&format!("gpu_mul_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::mul(black_box(&a), black_box(&b))).unwrap();
                });
            });

            group.bench_function(&format!("gpu_div_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::div(black_box(&a), black_box(&b))).unwrap();
                });
            });

            group.bench_function(&format!("gpu_sub_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::sub(black_box(&a), black_box(&b))).unwrap();
                });
            });

            group.bench_function(&format!("gpu_relu_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::relu(black_box(&a))).unwrap();
                });
            });
        }
    }
    group.finish();
}

fn bench_transpose(c: &mut Criterion) {
    let shapes = [
        ([256, 256], "256"),
        ([1024, 1024], "1024"),
        ([4096, 4096], "4096"),
    ];

    let mut group = c.benchmark_group("transpose");

    for (shape, size) in shapes.iter() {
        // CPU benchmarks
        {
            let a = create_random_cpu(shape);
            group.bench_function(format!("cpu_transpose_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::transpose(black_box(&a))).unwrap();
                });
            });
        }

        // GPU benchmarks
        #[cfg(feature = "cuda")]
        {
            init_context(0).unwrap();
            let _guard = CudaContextGuard::new().unwrap();

            let a = create_random_cuda(shape);
            group.bench_function(&format!("gpu_transpose_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::transpose(black_box(&a))).unwrap();
                });
            });
        }
    }
    group.finish();
}

fn bench_reductions(c: &mut Criterion) {
    let shapes = [
        ([256, 256], "256"),
        ([1024, 1024], "1024"),
        ([4096, 4096], "4096"),
    ];

    let mut group = c.benchmark_group("reductions");

    for (shape, size) in shapes.iter() {
        // CPU benchmarks
        {
            let a = create_random_cpu(shape);

            // Global reductions
            group.bench_function(format!("cpu_sum_global_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::sum(black_box(&a), None)).unwrap();
                });
            });

            group.bench_function(format!("cpu_mean_global_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::mean(black_box(&a), None)).unwrap();
                });
            });

            // Axis reductions
            group.bench_function(format!("cpu_sum_axis0_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::sum(black_box(&a), Some(0))).unwrap();
                });
            });

            group.bench_function(format!("cpu_mean_axis0_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::mean(black_box(&a), Some(0))).unwrap();
                });
            });
        }

        // GPU benchmarks
        #[cfg(feature = "cuda")]
        {
            init_context(0).unwrap();
            let _guard = CudaContextGuard::new().unwrap();

            let a = create_random_cuda(shape);

            // Global reductions
            group.bench_function(&format!("gpu_sum_global_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::sum(black_box(&a), None)).unwrap();
                });
            });

            group.bench_function(&format!("gpu_mean_global_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::mean(black_box(&a), None)).unwrap();
                });
            });

            // Axis reductions
            group.bench_function(&format!("gpu_sum_axis0_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::sum(black_box(&a), Some(0))).unwrap();
                });
            });

            group.bench_function(&format!("gpu_mean_axis0_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::mean(black_box(&a), Some(0))).unwrap();
                });
            });
        }
    }
    group.finish();
}

fn bench_nn_ops(c: &mut Criterion) {
    let shapes = [
        ([256, 10], "256x10"),
        ([1024, 10], "1024x10"),
        ([4096, 10], "4096x10"),
    ];

    let mut group = c.benchmark_group("nn_ops");

    for (shape, size) in shapes.iter() {
        // CPU benchmarks
        {
            let a = create_random_cpu(shape);

            group.bench_function(format!("cpu_log_softmax_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::log_softmax(black_box(&a), 1)).unwrap();
                });
            });
        }

        // GPU benchmarks
        #[cfg(feature = "cuda")]
        {
            init_context(0).unwrap();
            let _guard = CudaContextGuard::new().unwrap();

            let a = create_random_cuda(shape);

            group.bench_function(&format!("gpu_log_softmax_{}", size), |bencher| {
                bencher.iter(|| {
                    black_box(ops::log_softmax(black_box(&a), 1)).unwrap();
                });
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_matrix_multiply,
    bench_element_wise,
    bench_transpose,
    bench_reductions,
    bench_nn_ops
);
criterion_main!(benches);
