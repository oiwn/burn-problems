use crate::data::RawFeatures;
use crate::scaler::StandardScaler;
use rand::{seq::SliceRandom, thread_rng};

#[derive(Debug, Clone)]
pub struct PreparedFeatures {
    pub features: Vec<f32>, // Normalized/scaled features
    #[allow(dead_code)]
    pub label: Option<i64>, // 0 or 1
}

impl PreparedFeatures {
    pub fn from_raw(raw: &RawFeatures, scaler: Option<&StandardScaler>) -> Self {
        let features = raw.as_vec();

        // Scale if scaler provided
        let scaled_features = if let Some(scaler) = scaler {
            scaler.transform(&features)
        } else {
            features
        };

        Self {
            features: scaled_features,
            label: raw.get_label(),
        }
    }

    /// Split dataset into train and test parts
    pub fn split_dataset(data: Vec<Self>, train_ratio: f32) -> (Vec<Self>, Vec<Self>) {
        let mut rng = thread_rng();
        let mut data = data; // make the vector mutable

        // Shuffle the dataset
        data.shuffle(&mut rng);

        // Calculate split index
        let split_index = (data.len() as f32 * train_ratio).round() as usize;

        // Split the data
        let train_data = data[..split_index].to_vec();
        let test_data = data[split_index..].to_vec();

        (train_data, test_data)
    }
}
