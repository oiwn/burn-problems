mod data;
mod dataset;
mod model;
mod prepared;
mod scaler;
mod training;
mod utils;

use crate::prepared::PreparedFeatures;
use crate::utils::load_training_data;
use clap::{Parser, Subcommand};
use statrs::statistics::Statistics;

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Train,
    TestTrainingData,
    TestBCELoss,
    TestLabelPercentage,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Train) => {
            model_train();
        }
        Some(Commands::TestTrainingData) => {
            test_training_data();
        }
        Some(Commands::TestBCELoss) => {
            test_bce_loss();
        }
        Some(Commands::TestLabelPercentage) => {
            let (train_data, test_data) = load_training_data().unwrap();
            check_label_distribution(train_data.as_slice(), "Train");
            check_label_distribution(test_data.as_slice(), "Test");
        }
        _ => {
            eprintln!("Wrong command");
        }
    }
}

pub fn model_train() {
    use crate::model::DemoClassifierModelConfig;
    use crate::training::{train, TrainingConfig};
    use burn::optim::AdamConfig;

    type Backend = burn::backend::Wgpu;
    type AutoDiffBackend = burn::backend::Autodiff<Backend>;

    println!("Loading things to train DemoModel");

    let device = burn::backend::wgpu::WgpuDevice::default();
    let (train_data, test_data) = load_training_data().unwrap();

    for item in train_data[0..5].iter() {
        println!("Train data: {:?}", item);
    }

    for item in test_data[0..5].iter() {
        println!("Test data: {:?}", item);
    }

    println!(
        "Device: {:?} Train set: {}, Test set: {}",
        device,
        train_data.len(),
        test_data.len()
    );

    train::<AutoDiffBackend>(
        train_data,
        test_data,
        "data/burn_models",
        TrainingConfig::new(DemoClassifierModelConfig::new(), AdamConfig::new()),
        device,
    );
    println!("Training complete!");
}

pub fn test_training_data() {
    println!("=== Training Data Check (Rust) ===");
    match load_training_data() {
        Ok((train_data, test_data)) => {
            println!("\nDataset sizes:");
            println!("Train: {} Test: {}", train_data.len(), test_data.len());

            // Calculate statistics for first 3 features
            println!("\nFeature statistics (first 3):");
            for feature_idx in 0..3 {
                let train_feature: Vec<f64> = train_data
                    .iter()
                    .map(|x| x.features[feature_idx] as f64)
                    .collect();

                println!("\nFeature {}:", feature_idx);
                println!("Mean:   {:.4}", train_feature.clone().mean());
                println!("StdDev: {:.4}", train_feature.clone().std_dev());
                println!("Min:    {:.4}", train_feature.clone().min());
                println!("Max:    {:.4}", train_feature.clone().max());
            }

            println!("\nSample values (first 3 samples, first 3 features):");
            for (i, sample) in train_data.iter().take(3).enumerate() {
                println!("Sample {}: {:?}", i, &sample.features[..3]);
            }
        }
        Err(e) => println!("Error loading data: {}", e),
    }
}

fn test_bce_loss() {
    use burn::backend::ndarray::NdArray;
    use burn::nn::loss::BinaryCrossEntropyLossConfig;
    use burn::tensor::{Int, Tensor};

    type B = NdArray;
    let device = Default::default();

    // Initialize loss function
    let bce_loss = BinaryCrossEntropyLossConfig::new().init(&device);

    // Helper function to format tensor data
    fn format_tensor_data<const D: usize>(tensor: &Tensor<B, D>) -> String {
        let data: Vec<f32> = tensor.to_data().to_vec().unwrap();
        format!(
            "[{}]",
            data.iter()
                .map(|x| format!("{:.4}", x))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    fn format_int_tensor_data<const D: usize>(tensor: &Tensor<B, D, Int>) -> String {
        let data: Vec<i64> = tensor.to_data().to_vec().unwrap();
        format!(
            "[{}]",
            data.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    // Test Case 1: Perfect predictions
    let pred_data: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0];
    let target_data: Vec<u8> = vec![1, 0, 1, 0];
    let predictions = Tensor::<B, 1>::from_floats(pred_data.as_slice(), &device);
    let targets = Tensor::<B, 1, Int>::from_ints(target_data.as_slice(), &device);

    let loss = bce_loss.forward(predictions.clone(), targets.clone());

    println!("\nTest Case 1 - Perfect predictions:");
    println!("Predictions: tensor({})", format_tensor_data(&predictions));
    println!("Targets:     tensor({})", format_int_tensor_data(&targets));
    println!("Loss:        {:.8}", loss.into_scalar());

    // Test Case 2: Completely wrong predictions
    let pred_data: Vec<f32> = vec![0.0, 1.0, 0.0, 1.0];
    let predictions = Tensor::<B, 1>::from_floats(pred_data.as_slice(), &device);
    let loss = bce_loss.forward(predictions.clone(), targets.clone());

    println!("\nTest Case 2 - Wrong predictions:");
    println!("Predictions: tensor({})", format_tensor_data(&predictions));
    println!("Targets:     tensor({})", format_int_tensor_data(&targets));
    println!("Loss:        {:.8}", loss.into_scalar());

    // Test Case 3: Uncertain predictions (0.5)
    let pred_data: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5];
    let predictions = Tensor::<B, 1>::from_floats(pred_data.as_slice(), &device);
    let loss = bce_loss.forward(predictions.clone(), targets.clone());

    println!("\nTest Case 3 - Uncertain predictions:");
    println!("Predictions: tensor({})", format_tensor_data(&predictions));
    println!("Targets:     tensor({})", format_int_tensor_data(&targets));
    println!("Loss:        {:.8}", loss.into_scalar());
}

/// Display distribution of binary labels (0/1) in a Vec of PreparedPageFeatures.
pub fn check_label_distribution(data: &[PreparedFeatures], name: &str) {
    let total = data.len();
    if total == 0 {
        println!("{} dataset is empty!", name);
        return;
    }

    let ones = data.iter().filter(|x| x.label.unwrap_or(0) == 1).count();
    let zeroes = total - ones;

    let pct_ones = (ones as f64 / total as f64) * 100.0;
    let pct_zeroes = (zeroes as f64 / total as f64) * 100.0;

    println!(
        "{} distribution: total={}, zeroes={} ({:.1}%), ones={} ({:.1}%)",
        name, total, zeroes, pct_zeroes, ones, pct_ones
    );
}
