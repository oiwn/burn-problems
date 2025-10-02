// use crate::is_page_model::features::PageFeatures;
use burn::{
    nn::{Linear, LinearConfig, Relu, Sigmoid},
    prelude::*,
};

/// Model for binary classification of articles
/// input->64->32->1
#[derive(Module, Debug)]
pub struct DemoClassifierModel<B: Backend> {
    input_layer: Linear<B>,
    hidden_layer1: Linear<B>,
    output_layer: Linear<B>,
    activation: Relu,
    sigmoid: Sigmoid,
}

/// Configuration for the article classifier model
#[derive(Config, Debug)]
pub struct DemoClassifierModelConfig {
    #[config(default = 20)] // Must match number of features
    pub input_size: usize,
    #[config(default = 64)] // First hidden layer size
    pub hidden_size1: usize,
    #[config(default = 32)] // Second hidden layer size
    pub hidden_size2: usize,
    #[config(default = 1)] // Binary classification output
    pub output_size: usize,
}

impl DemoClassifierModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DemoClassifierModel<B> {
        DemoClassifierModel {
            // Input layer -> First hidden layer (20 -> 64)
            input_layer: LinearConfig::new(self.input_size, self.hidden_size1).init(device),
            // First hidden layer -> Second hidden layer (64 -> 32)
            hidden_layer1: LinearConfig::new(self.hidden_size1, self.hidden_size2).init(device),
            // Second hidden layer -> Output (32 -> 1)
            output_layer: LinearConfig::new(self.hidden_size2, self.output_size).init(device),
            activation: Relu::new(),
            sigmoid: Sigmoid::new(),
        }
    }
}

impl<B: Backend> DemoClassifierModel<B> {
    /// Forward pass of the model, matching PyTorch implementation:
    /// x = self.relu(self.fc1(x))
    /// x = self.relu(self.fc2(x))
    /// x = self.sigmoid(self.fc3(x))
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // First layer with ReLU
        let x = self.input_layer.forward(x);
        let x = self.activation.forward(x);

        // Hidden layer with ReLU
        let x = self.hidden_layer1.forward(x);
        let x = self.activation.forward(x);

        // Output layer - return logits (no sigmoid, BCE loss will handle it)
        self.output_layer.forward(x)
    }

    /// Forward pass with sigmoid for inference
    pub fn forward_inference(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let logits = self.forward(x);
        self.sigmoid.forward(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::tensor::Tensor;

    type TestBackend = burn::backend::ndarray::NdArray;

    #[test]
    fn test_model_forward() {
        let device = NdArrayDevice::Cpu;

        // Initialize model with test config
        let config = DemoClassifierModelConfig {
            input_size: 20,
            hidden_size1: 64,
            hidden_size2: 32,
            output_size: 1,
        };
        let model = config.init::<TestBackend>(&device);

        // Create test input tensor (batch_size = 2, features = 20)
        let input = Tensor::<TestBackend, 2>::zeros([2, 20], &device);

        // Run forward pass
        let output = model.forward(input);

        // Check output shape
        assert_eq!(
            output.dims(),
            [2, 1],
            "Output shape should be [batch_size, 1]"
        );

        // Check output values are between 0 and 1 (due to sigmoid)
        let output_data: Vec<f32> = output.to_data().to_vec().unwrap();
        for val in output_data {
            assert!(
                (0.0..=1.0).contains(&val),
                "Output values should be between 0 and 1"
            );
        }
    }

    #[test]
    fn test_model_initialization() {
        let device = NdArrayDevice::Cpu;

        let config = DemoClassifierModelConfig {
            input_size: 20,
            hidden_size1: 64,
            hidden_size2: 32,
            output_size: 1,
        };

        let model = config.init::<TestBackend>(&device);

        // Verify layer dimensions through forward pass with test data
        let input = Tensor::<TestBackend, 2>::zeros([1, 20], &device);
        let output = model.forward(input);

        assert_eq!(
            output.dims(),
            [1, 1],
            "Single sample should produce single output"
        );
    }
}
