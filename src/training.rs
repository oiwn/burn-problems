use crate::dataset::{DemoBatch, DemoBatcher, DemoDataset};
use crate::model::{DemoClassifierModel, DemoClassifierModelConfig};
use crate::prepared::PreparedFeatures;
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    grad_clipping::GradientClippingConfig,
    module::Module,
    nn::loss::BinaryCrossEntropyLossConfig,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Int, Tensor,
    },
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
    },
};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: DemoClassifierModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 20)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 1)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 0.0001)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

impl<B: Backend> DemoClassifierModel<B> {
    pub fn forward_classification(
        &self,
        features: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(features); // [batch_size, 1]
        let loss = BinaryCrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone().squeeze(), targets.clone());

        // let output = self.forward(features);
        // let batch_size = output.dims()[0];
        // let loss = BinaryCrossEntropyLossConfig::new()
        //     .init(&output.device())
        //     .forward(output.clone(), targets.clone().reshape([batch_size, 1]));

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<DemoBatch<B>, ClassificationOutput<B>>
    for DemoClassifierModel<B>
{
    fn step(&self, batch: DemoBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.features, batch.labels);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<DemoBatch<B>, ClassificationOutput<B>> for DemoClassifierModel<B> {
    fn step(&self, batch: DemoBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.features, batch.labels)
    }
}

pub fn train<B: AutodiffBackend>(
    train_data: Vec<PreparedFeatures>,
    test_data: Vec<PreparedFeatures>,
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
) {
    // Setup artifact directory and save config
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    // Set random seed
    B::seed(&device, config.seed);

    // Create batchers for train and validation
    let batcher_train = DemoBatcher::<B>::new(device.clone());
    let batcher_valid = DemoBatcher::<B::InnerBackend>::new(device.clone());

    // Create dataloaders
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(DemoDataset::train(train_data));

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(DemoDataset::test(test_data));

    // Initialize BCE loss
    // let loss_fn = BinaryCrossEntropyLossConfig::new().init(&device);

    // Create model
    let model = config.model.init::<B>(&device);

    // Get Optimizer
    let optimizer = config
        .optimizer
        .init::<B, DemoClassifierModel<B>>()
        .with_grad_clipping(GradientClippingConfig::Norm(1.0).init());

    // Initialize learner
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, optimizer, config.learning_rate);

    // Train model
    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    // Save trained model
    model_trained
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
