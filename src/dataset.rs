use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};

use crate::prepared::PreparedFeatures;

// Dataset struct to hold our training/test data
pub struct DemoDataset {
    features: Vec<PreparedFeatures>,
    #[allow(dead_code)]
    is_training: bool,
}

impl DemoDataset {
    pub fn train(features: Vec<PreparedFeatures>) -> Self {
        Self {
            features,
            is_training: true,
        }
    }

    pub fn test(features: Vec<PreparedFeatures>) -> Self {
        Self {
            features,
            is_training: false,
        }
    }
}

// Implementation of Dataset trait for ArticleDataset
impl Dataset<PreparedFeatures> for DemoDataset {
    fn get(&self, index: usize) -> Option<PreparedFeatures> {
        self.features.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.features.len()
    }
}

// Batch struct to hold tensors for training
#[derive(Clone, Debug)]
pub struct DemoBatch<B: Backend> {
    pub features: Tensor<B, 2>,    // [batch_size, num_features]
    pub labels: Tensor<B, 1, Int>, // [batch_size]
}

// Batcher implementation to convert data into tensors
#[derive(Clone)]
pub struct DemoBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> DemoBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<B, PreparedFeatures, DemoBatch<B>> for DemoBatcher<B> {
    fn batch(&self, items: Vec<PreparedFeatures>, _device: &B::Device) -> DemoBatch<B> {
        let batch_size = items.len();
        let num_features = items[0].features.len();

        // Prepare feature data
        let mut features_data: Vec<f32> = Vec::with_capacity(batch_size * num_features);
        let mut labels_data = Vec::with_capacity(batch_size);

        for item in items {
            features_data.extend(item.features.iter());
            // label is Option<i64>
            labels_data.push(item.label.unwrap_or(0));
        }

        let features_data = TensorData::new(features_data, [batch_size, num_features]);
        let labels_data = TensorData::new(labels_data, [batch_size]);

        // Create tensors
        let features = Tensor::from_data(features_data, &self.device);
        let labels = Tensor::from_data(labels_data, &self.device);

        DemoBatch { features, labels }
    }
}

// Example usage in training setup
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::load_training_data;
    use approx::assert_relative_eq;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::data::dataloader::DataLoaderBuilder;

    type TestBackend = burn::backend::ndarray::NdArray;

    #[test]
    fn test_dataloader_setup() {
        let device = NdArrayDevice::Cpu;

        // Create some dummy data
        let features = vec![
            PreparedFeatures {
                features: vec![0.1, 0.2, 0.3, 0.4, 0.5], // example features
                label: Some(1),                          // example label
            },
            PreparedFeatures {
                features: vec![0.2, 0.3, 0.4, 0.5, 0.6],
                label: Some(0),
            },
        ];

        // Create dataset
        let dataset = DemoDataset::train(features);

        // Create batcher
        let batcher = DemoBatcher::<TestBackend>::new(device);

        // Create dataloader
        let dataloader = DataLoaderBuilder::new(batcher)
            .batch_size(2)
            .shuffle(42)
            .num_workers(1)
            .build(dataset);

        // Dataloader is ready for training
        assert!(dataloader.num_items() > 0);

        // Test batch shape
        let batch = dataloader.iter().next().unwrap();
        assert_eq!(batch.features.dims(), [2, 5]);
        assert_eq!(batch.labels.dims(), [2]);
    }

    #[test]
    fn test_batcher_tensor_creation() {
        let device = NdArrayDevice::Cpu;
        let batcher = DemoBatcher::<TestBackend>::new(device);

        // Create test data
        let batch_items = vec![
            PreparedFeatures {
                features: vec![0.1, 0.2, 0.3],
                label: Some(1),
            },
            PreparedFeatures {
                features: vec![0.4, 0.5, 0.6],
                label: Some(0),
            },
        ];

        // Batch the data
        let batch = batcher.batch(batch_items, &device);

        // Print shapes and data
        println!("Features tensor shape: {:?}", batch.features.dims());
        println!("Labels tensor shape: {:?}", batch.labels.dims());

        // Print actual tensor data
        println!("Features: {}", batch.features);
        println!("Labels: {}", batch.labels);
    }

    #[test]
    fn test_batch_shapes_and_values() {
        let device = NdArrayDevice::Cpu;
        let batcher = DemoBatcher::<TestBackend>::new(device);

        // Create test data
        let test_features = [
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];

        let batch_items = test_features
            .iter()
            .map(|features| PreparedFeatures {
                features: features.clone(),
                label: Some(1),
            })
            .collect::<Vec<_>>();

        // Batch the data
        let batch = batcher.batch(batch_items, &device);

        // Check tensor shapes
        assert_eq!(
            batch.features.dims(),
            [3, 3],
            "Features shape should be [batch_size, num_features]"
        );
        assert_eq!(
            batch.labels.dims(),
            [3],
            "Labels shape should be [batch_size]"
        );

        // Verify values
        let features_data: Vec<f32> = batch.features.to_data().to_vec().unwrap();
        let labels_data: Vec<i64> = batch.labels.to_data().to_vec().unwrap();

        // Check some values
        assert_relative_eq!(features_data[0], 0.1, epsilon = 1e-6);
        assert_relative_eq!(features_data[4], 0.5, epsilon = 1e-6);
        assert_relative_eq!(features_data[8], 0.9, epsilon = 1e-6);

        assert_eq!(labels_data[0], 1);
        assert_eq!(labels_data[1], 1);
        assert_eq!(labels_data[2], 1);
    }

    #[test]
    fn test_dataloader_batch_size() {
        let device = NdArrayDevice::Cpu;

        // Create test data - make sure we have more than batch_size
        let test_data = (0..50)
            .map(|i| PreparedFeatures {
                features: vec![i as f32; 3],
                label: Some(if i % 2 == 0 { 1 } else { 0 }),
            })
            .collect::<Vec<_>>();

        // Create dataset
        let dataset = DemoDataset::train(test_data);

        // Create dataloader with specific batch size
        let batch_size = 16;
        let batcher = DemoBatcher::<TestBackend>::new(device);

        let dataloader = DataLoaderBuilder::new(batcher)
            .batch_size(batch_size)
            .shuffle(42)
            .num_workers(1)
            .build(dataset);

        // Get first batch and verify its size
        let first_batch = dataloader.iter().next().unwrap();
        assert_eq!(
            first_batch.features.dims()[0],
            batch_size,
            "First dimension (batch size) should match configured batch size"
        );
        assert_eq!(
            first_batch.labels.dims()[0],
            batch_size,
            "Labels batch size should match configured batch size"
        );
    }

    #[test]
    fn test_full_dataset_loading_and_batching() {
        // Load the actual training data
        let (train_data, test_data) = load_training_data().unwrap();
        println!(
            "Loaded train: {}, test: {} samples",
            train_data.len(),
            test_data.len()
        );

        let device = NdArrayDevice::Cpu;
        let batch_size = 32;
        let batcher = DemoBatcher::<TestBackend>::new(device);

        // Create dataloaders
        let train_loader = DataLoaderBuilder::new(batcher.clone())
            .batch_size(batch_size)
            .shuffle(42)
            .num_workers(1)
            .build(DemoDataset::train(train_data));

        // Check first few batches
        for (i, batch) in train_loader.iter().take(3).enumerate() {
            println!(
                "Batch {}: features shape: {:?}, labels shape: {:?}",
                i,
                batch.features.dims(),
                batch.labels.dims()
            );

            // Verify batch sizes
            assert_eq!(
                batch.features.dims()[0],
                batch_size,
                "Features batch size mismatch"
            );
            assert_eq!(
                batch.labels.dims()[0],
                batch_size,
                "Labels batch size mismatch"
            );

            // Print some statistics about the batch
            let labels_data: Vec<i64> = batch.labels.to_data().to_vec().unwrap();
            let pos_count = labels_data.iter().filter(|&&x| x == 1).count();
            let neg_count = labels_data.iter().filter(|&&x| x == 0).count();
            println!(
                "Batch {} distribution - positive: {}, negative: {}",
                i, pos_count, neg_count
            );
        }
    }

    #[test]
    fn test_data_distribution() {
        let (train_data, _test_data) = load_training_data().unwrap();

        // Check train data distribution
        let train_pos = train_data.iter().filter(|x| x.label.unwrap() == 1).count();
        let train_neg = train_data.len() - train_pos;

        println!("Train data distribution:");
        println!(
            "Positive: {} ({:.2}%)",
            train_pos,
            (train_pos as f32 / train_data.len() as f32) * 100.0
        );
        println!(
            "Negative: {} ({:.2}%)",
            train_neg,
            (train_neg as f32 / train_data.len() as f32) * 100.0
        );

        // Check feature statistics
        let num_features = train_data[0].features.len();
        for i in 0..num_features {
            let values: Vec<f32> = train_data.iter().map(|x| x.features[i]).collect();

            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let std = (values.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                / values.len() as f32)
                .sqrt();

            println!("Feature {}: mean = {:.3}, std = {:.3}", i, mean, std);
        }

        // Check for any NaN or infinity values
        let has_invalid = train_data
            .iter()
            .any(|x| x.features.iter().any(|&v| v.is_nan() || v.is_infinite()));

        assert!(
            !has_invalid,
            "Found NaN or infinity values in training data"
        );
    }
}
