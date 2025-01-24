import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def test_training_data(data_file='data/demo_training_data.csv'):
    print("=== Training Data Check (Python) ===")
    
    # Load and prepare data
    df = pd.read_csv(data_file)
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    _X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print("\nDataset sizes:")
    print(f"Train: {len(X_train)} Test: {len(X_test)}")
    
    # Calculate statistics for first 3 features
    print("\nFeature statistics (first 3):")
    for i in range(3):
        feature_name = X.columns[i]
        print(f"\nFeature {i} ({feature_name}):")
        print(f"Mean:   {X_train[:,i].mean():.4f}")
        print(f"StdDev: {X_train[:,i].std():.4f}")
        print(f"Min:    {X_train[:,i].min():.4f}")
        print(f"Max:    {X_train[:,i].max():.4f}")
    
    print("\nSample values (first 3 samples, first 3 features):")
    for i in range(3):
        print(f"Sample {i}: {X_train[i,:3]}")

if __name__ == "__main__":
    test_training_data()
  
