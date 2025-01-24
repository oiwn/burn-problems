//! Features manupulations
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawFeatures {
    pub feature1: usize,
    pub feature2: usize,
    pub feature3: usize,
    pub feature4: usize,
    pub feature5: bool,
    pub feature6: usize,
    pub feature7: usize,
    pub feature8: usize,
    pub feature9: f32,
    pub feature10: usize,
    pub feature11: usize,
    pub feature12: usize,
    // additional
    pub feature13: usize,
    pub feature14: usize,
    pub feature15: usize,
    pub feature16: usize,
    pub feature17: usize,
    // feature interactions
    pub feature18: f32,
    pub feature19: f32,
    pub feature20: f32,
    // label is optional
    pub label: Option<bool>,
}

impl RawFeatures {
    pub fn as_vec(&self) -> Vec<f32> {
        let feature5 = if self.feature5 { 1.0 } else { 0.0 };
        vec![
            self.feature1 as f32,
            self.feature2 as f32,
            self.feature3 as f32,
            self.feature4 as f32,
            feature5,
            self.feature6 as f32,
            self.feature7 as f32,
            self.feature8 as f32,
            self.feature9,
            self.feature10 as f32,
            self.feature11 as f32,
            self.feature12 as f32,
            // additional
            self.feature13 as f32,
            self.feature14 as f32,
            self.feature15 as f32,
            self.feature16 as f32,
            self.feature17 as f32,
            // feature interactions
            self.feature18,
            self.feature19,
            self.feature20,
        ]
    }

    pub fn get_label(&self) -> Option<i64> {
        self.label.map(|l| if l { 1 } else { 0 })
    }
}
