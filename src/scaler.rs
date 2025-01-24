use std::iter::Iterator;

#[derive(Debug, Default)]
pub struct StandardScaler {
    pub means: Vec<f32>,
    pub stds: Vec<f32>,
    fitted: bool,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            means: Vec::new(),
            stds: Vec::new(),
            fitted: false,
        }
    }

    pub fn fit(&mut self, data: &[Vec<f32>]) {
        if data.is_empty() {
            return;
        }

        let n_features = data[0].len();
        self.means = vec![0.0; n_features];
        self.stds = vec![0.0; n_features];

        // Calculate means
        let n_samples = data.len() as f32;
        for sample in data {
            for (i, &value) in sample.iter().enumerate() {
                self.means[i] += value;
            }
        }
        for mean in &mut self.means {
            *mean /= n_samples;
        }

        // Calculate standard deviations
        for sample in data {
            for (i, &value) in sample.iter().enumerate() {
                self.stds[i] += (value - self.means[i]).powi(2);
            }
        }
        for std in &mut self.stds {
            *std = (*std / n_samples).sqrt();
            // Prevent division by zero
            if *std == 0.0 {
                *std = 1.0;
            }
        }

        self.fitted = true;
    }

    pub fn transform(&self, features: &[f32]) -> Vec<f32> {
        if !self.fitted {
            panic!("Scaler must be fitted before transform");
        }

        features
            .iter()
            .zip(self.means.iter().zip(self.stds.iter()))
            .map(|(&x, (&mean, &std))| (x - mean) / std)
            .collect()
    }

    #[allow(dead_code)]
    pub fn fit_transform(&mut self, data: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.fit(data);
        data.iter()
            .map(|features| self.transform(features))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_standard_scaler_single_feature() {
        let data = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];

        let mut scaler = StandardScaler::new();
        scaler.fit(&data);

        assert_relative_eq!(scaler.means[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(scaler.stds[0], std::f32::consts::SQRT_2, epsilon = 1e-6);

        let transformed = scaler.transform(&[1.0]);
        assert_relative_eq!(transformed[0], -std::f32::consts::SQRT_2, epsilon = 1e-6);
    }

    #[test]
    fn test_standard_scaler_multiple_features() {
        let data = vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]];

        let mut scaler = StandardScaler::new();
        let transformed = scaler.fit_transform(&data);

        // Check means
        assert_relative_eq!(scaler.means[0], 2.0);
        assert_relative_eq!(scaler.means[1], 5.0);

        // Check first transformed row
        assert_relative_eq!(transformed[0][0], -1.2247448, epsilon = 1e-6);
        assert_relative_eq!(transformed[0][1], -1.2247448, epsilon = 1e-6);
    }

    #[test]
    fn test_constant_feature() {
        let data = vec![vec![2.0, 1.0], vec![2.0, 2.0], vec![2.0, 3.0]];

        let mut scaler = StandardScaler::new();
        let transformed = scaler.fit_transform(&data);

        // Constant feature should become zero
        assert_relative_eq!(transformed[0][0], 0.0);
        assert_relative_eq!(transformed[1][0], 0.0);
        assert_relative_eq!(transformed[2][0], 0.0);
    }

    #[test]
    #[should_panic(expected = "Scaler must be fitted before transform")]
    fn test_transform_without_fit() {
        let scaler = StandardScaler::new();
        scaler.transform(&[1.0, 2.0]);
    }
}
