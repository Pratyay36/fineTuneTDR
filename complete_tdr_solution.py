import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

# Define the model class
class TDRAnalyzer:
    def __init__(self):
        self.ports = ['TDRDF1', 'TDRDF2', 'TDRDF3', 'TDRDF4']
        self.models = {}
        self.scalers = {}
    
    def create_features(self, pitch, time):
        """Create features from pitch and time"""
        features = {
            'pitch': pitch,
            'time': time,
            'time_sq': time ** 2,
            'time_cu': time ** 3,
            'time_sqrt': np.sqrt(max(time, 1e-6)),
            'pitch_time': pitch * time,
            'time_exp': np.exp(-time),
            'time_log': np.log(max(time, 1e-6)),
            'time_sin': np.sin(2 * np.pi * time),
            'time_cos': np.cos(2 * np.pi * time)
        }
        return np.array([list(features.values())])
    
    def train_port(self, df, port_name):
        """Train model for a specific port"""
        print(f"\n🔄 Training {port_name}...")
        
        X = df[['Pitch', 'Time']].values
        y = df[port_name].values
        
        # Create feature matrix
        X_features = []
        for pitch, time in X:
            X_features.append(self.create_features(pitch, time)[0])
        X_features = np.array(X_features)
        
        # Split data
        split_idx = int(len(X_features) * 0.8)
        X_train, X_test = X_features[:split_idx], X_features[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)
        
        gb_model = HistGradientBoostingRegressor(max_depth=10, random_state=42)
        gb_model.fit(X_train_scaled, y_train)
        
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train_scaled, y_train)
        
        # Ensemble predictions (simple average)
        rf_pred = rf_model.predict(X_test_scaled)
        gb_pred = gb_model.predict(X_test_scaled)
        ridge_pred = ridge_model.predict(X_test_scaled)
        
        ensemble_pred = (rf_pred + gb_pred + ridge_pred) / 3
        
        # Calculate metrics
        r2 = r2_score(y_test, ensemble_pred)
        mae = mean_absolute_error(y_test, ensemble_pred)
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        
        print(f"  ✅ R²: {r2:.4f}, MAE: {mae:.2f}Ω, RMSE: {rmse:.2f}Ω")
        
        # Store models and scaler
        self.models[port_name] = {
            'rf': rf_model,
            'gb': gb_model,
            'ridge': ridge_model
        }
        self.scalers[port_name] = scaler
        
        return r2, mae, rmse
    
    def train_all(self, csv_file):
        """Train models for all ports"""
        print("="*60)
        print("TRAINING TDR PREDICTION MODELS")
        print("="*60)
        
        # Load data
        df = pd.read_csv(csv_file)
        print(f"\n✅ Loaded {len(df)} rows from {csv_file}")
        
        results = {}
        for port in self.ports:
            r2, mae, rmse = self.train_port(df, port)
            results[port] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
        
        # Save models
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        joblib.dump(self, 'tdr_analyzer.pkl')
        print("✅ Model saved as 'tdr_analyzer.pkl'")
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        results_df = pd.DataFrame(results).T
        print(results_df.round(4))
        
        return results
    
    def predict(self, port_name, pitch, time):
        """Predict TDR value for a given port, pitch, and time"""
        if port_name not in self.models:
            raise ValueError(f"Port {port_name} not trained")
        
        # Create features
        X = self.create_features(pitch, time)
        
        # Scale
        X_scaled = self.scalers[port_name].transform(X)
        
        # Get predictions from each model
        rf_pred = self.models[port_name]['rf'].predict(X_scaled)[0]
        gb_pred = self.models[port_name]['gb'].predict(X_scaled)[0]
        ridge_pred = self.models[port_name]['ridge'].predict(X_scaled)[0]
        
        # Ensemble (simple average)
        prediction = (rf_pred + gb_pred + ridge_pred) / 3
        
        return prediction
    
    def predict_waveform(self, pitch, time_range, num_points=50):
        """Generate complete TDR waveform for a given pitch"""
        times = np.linspace(time_range[0], time_range[1], num_points)
        
        results = {'Time_ns': times}
        for port in self.ports:
            predictions = [self.predict(port, pitch, t) for t in times]
            results[port] = predictions
        
        return pd.DataFrame(results)
    
    def predict_batch(self, input_csv, output_csv):
        """Make predictions for multiple cases from a CSV file"""
        df_input = pd.read_csv(input_csv)
        
        predictions = []
        for _, row in df_input.iterrows():
            pred_row = {'pitch': row['pitch'], 'time': row['time']}
            for port in self.ports:
                pred_row[f'pred_{port}'] = self.predict(port, row['pitch'], row['time'])
            predictions.append(pred_row)
        
        df_output = pd.DataFrame(predictions)
        df_output.to_csv(output_csv, index=False)
        return df_output

def main():
    print("\n" + "="*60)
    print("TDR PREDICTION SYSTEM")
    print("="*60)
    
    # Check if model exists, if not train it
    if os.path.exists('tdr_analyzer.pkl'):
        print("\n📂 Loading existing model...")
        analyzer = joblib.load('tdr_analyzer.pkl')
        print("✅ Model loaded successfully!")
    else:
        print("\n📂 No existing model found. Training new model...")
        analyzer = TDRAnalyzer()
        analyzer.train_all('tdr_data.csv')
    
    # Make sample predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    test_cases = [
        (3.3, 0.0),
        (3.3, 0.25),
        (3.3, 0.5),
        (3.3, 0.75),
        (3.3, 1.0),
    ]
    
    for pitch, time in test_cases:
        print(f"\n📊 Pitch={pitch}, Time={time}ns:")
        for port in analyzer.ports:
            pred = analyzer.predict(port, pitch, time)
            print(f"   {port}: {pred:.2f} Ω")
    
    # Generate full waveform
    print("\n" + "="*60)
    print("GENERATING FULL TDR WAVEFORM")
    print("="*60)
    
    waveform_df = analyzer.predict_waveform(pitch=3.3, time_range=(0, 1.1), num_points=30)
    waveform_df.to_csv('tdr_complete_waveform.csv', index=False)
    print("✅ Saved to 'tdr_complete_waveform.csv'")
    print("\nFirst 10 rows:")
    print(waveform_df.head(10))
    
    # Create batch prediction example
    print("\n" + "="*60)
    print("BATCH PREDICTIONS EXAMPLE")
    print("="*60)
    
    # Create sample input file
    sample_input = pd.DataFrame({
        'pitch': [3.3, 3.3, 3.3, 3.5, 3.0],
        'time': [0.1, 0.3, 0.7, 0.5, 0.4]
    })
    sample_input.to_csv('sample_input.csv', index=False)
    print("Created sample_input.csv")
    
    # Make batch predictions
    batch_predictions = analyzer.predict_batch('sample_input.csv', 'batch_predictions.csv')
    print("\nBatch Predictions:")
    print(batch_predictions.round(2))
    
    print("\n" + "="*60)
    print("✅ ALL DONE!")
    print("="*60)
    print("\nFiles created:")
    print("  - tdr_analyzer.pkl (trained model)")
    print("  - tdr_complete_waveform.csv (full TDR waveform)")
    print("  - sample_input.csv (example input)")
    print("  - batch_predictions.csv (batch predictions)")

if __name__ == "__main__":
    main()