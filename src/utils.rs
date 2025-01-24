//! Utilities to load / store data required to train model
use crate::data::RawFeatures;
use crate::prepared::PreparedFeatures;
use crate::scaler::StandardScaler;
use csv::ReaderBuilder;
use serde::Deserialize;
use std::fs::File;

// Load data from CSV
pub fn load_from_csv<T>(file_path: &str) -> anyhow::Result<Vec<T>>
where
    T: for<'de> Deserialize<'de>,
{
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new().from_reader(file);
    let mut data = Vec::new();

    for result in rdr.deserialize() {
        let record: T = result?;
        data.push(record);
    }
    Ok(data)
}

pub fn load_training_data() -> anyhow::Result<(Vec<PreparedFeatures>, Vec<PreparedFeatures>)> {
    // 1. Load raw data
    let raw_features: Vec<RawFeatures> = load_from_csv("data/demo_training_data.csv")?;

    // 2. Create and fit scaler
    let mut scaler = StandardScaler::new();
    scaler.fit(&raw_features.iter().map(|f| f.as_vec()).collect::<Vec<_>>());

    // 3. Convert all features with scaling
    let prepared_features: Vec<PreparedFeatures> = raw_features
        .iter()
        .map(|raw| PreparedFeatures::from_raw(raw, Some(&scaler)))
        .collect();

    // 4. Split dataset
    Ok(PreparedFeatures::split_dataset(prepared_features, 0.8))
}
