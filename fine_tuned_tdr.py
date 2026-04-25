import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

class FineTunedTDRModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.imputers = {}
        
    def create_features_safe(self, X):
        """Create advanced features WITHOUT generating NaN values"""
        X_new = X.copy()
        
        pitch = X['Pitch'].values
        time = X['Time'].values
        
        # Safe operations
        time_safe = np.maximum(time, 1e-6)
        
        # Polynomial features
        X_new['Time_squared'] = time ** 2
        X_new['Time_cubed'] = time ** 3
        X_new['Time_sqrt'] = np.sqrt(np.maximum(time, 0))
        
        # Interaction features
        X_new['Pitch_Time'] = pitch * time
        X_new['Pitch_Time_squared'] = pitch * (time ** 2)
        
        # Exponential features
        X_new['Time_exp'] = np.exp(-time)
        X_new['Time_exp_squared'] = np.exp(-(time ** 2))
        
        # Logarithmic features
        X_new['Time_log'] = np.log(np.maximum(time + 0.001, 0.001))
        
        # Sinusoidal features
        X_new['Time_sin'] = np.sin(2 * np.pi * time)
        X_new['Time_cos'] = np.cos(2 * np.pi * time)
        
        # Replace any NaN with 0
        X_new = X_new.fillna(0)
        X_new = X_new.replace([np.inf, -np.inf], 0)
        
        return X_new
    
    def train_ensemble_model(self, X_train, y_train):
        """Train ensemble of models"""
        
        print(f"  Training Random Forest...")
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_score = rf_model.score(X_train, y_train)
        
        print(f"  Training Gradient Boosting...")
        gb_model = HistGradientBoostingRegressor(max_depth=10, random_state=42)
        gb_model.fit(X_train, y_train)
        gb_score = gb_model.score(X_train, y_train)
        
        print(f"  Training Neural Network...")
        nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, verbose=False)
        nn_model.fit(X_train, y_train)
        nn_score = nn_model.score(X_train, y_train)
        
        print(f"  Training Ridge Regression...")
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train, y_train)
        ridge_score = ridge_model.score(X_train, y_train)
        
        # Store models with weights based on performance
        models_info = {
            'RandomForest': {'model': rf_model, 'weight': max(rf_score, 0.1)},
            'GradientBoosting': {'model': gb_model, 'weight': max(gb_score, 0.1)},
            'NeuralNetwork': {'model': nn_model, 'weight': max(nn_score, 0.1)},
            'Ridge': {'model': ridge_model, 'weight': max(ridge_score, 0.1)}
        }
        
        # Normalize weights
        total_weight = sum(info['weight'] for info in models_info.values())
        for info in models_info.values():
            info['weight'] = info['weight'] / total_weight
        
        return models_info
    
    def ensemble_predict(self, models_info, X):
        """Weighted average prediction"""
        predictions = []
        weights = []
        
        for model_info in models_info.values():
            pred = model_info['model'].predict(X)
            predictions.append(pred)
            weights.append(model_info['weight'])
        
        weights = np.array(weights).reshape(-1, 1)
        predictions = np.array(predictions)
        weighted_pred = np.average(predictions, axis=0, weights=weights.flatten())
        
        return weighted_pred
    
    def train_tdr_model(self, X, y, port_name):
        """Train model for a specific port"""
        
        print(f"\n{'='*60}")
        print(f"Training Model for {port_name}")
        print(f"{'='*60}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create features
        print("  Creating advanced features...")
        X_train_feat = self.create_features_safe(X_train)
        X_val_feat = self.create_features_safe(X_val)
        
        # Scale features
        print("  Scaling features...")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_feat)
        X_val_scaled = scaler.transform(X_val_feat)
        
        # Train ensemble
        print("  Training ensemble model...")
        models_info = self.train_ensemble_model(X_train_scaled, y_train)
        
        # Validate
        y_pred = self.ensemble_predict(models_info, X_val_scaled)
        
        # Calculate metrics
        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        
        print(f"\n  ✅ Validation Performance:")
        print(f"     R² Score: {r2:.4f}")
        print(f"     RMSE: {rmse:.4f} Ω")
        print(f"     MAE: {mae:.4f} Ω")
        
        # Store model
        self.models[port_name] = {
            'models': models_info,
            'scaler': scaler,
            'feature_engineer': self.create_features_safe
        }
        
        return self.models[port_name]
    
    def predict(self, port_name, pitch, time):
        """Make prediction"""
        
        if port_name not in self.models:
            raise ValueError(f"Model for {port_name} not found")
        
        model_info = self.models[port_name]
        
        # Create input
        X = pd.DataFrame({'Pitch': [pitch], 'Time': [time]})
        
        # Feature engineering
        X_feat = model_info['feature_engineer'](X)
        
        # Scale
        X_scaled = model_info['scaler'].transform(X_feat)
        
        # Predict
        prediction = self.ensemble_predict(model_info['models'], X_scaled)
        
        return prediction[0]

def main():
    print("="*60)
    print("FINE-TUNED TDR PREDICTION MODEL")
    print("="*60)
    
    # Check if CSV file exists
    csv_file = 'tdr_data.csv'
    
    if not os.path.exists(csv_file):
        print(f"\n❌ Error: '{csv_file}' not found!")
        print("\nPlease make sure your CSV file is named 'tdr_data.csv'")
        print("The CSV should have columns: Pitch, Time, TDRDF1, TDRDF2, TDRDF3, TDRDF4")
        return
    
    # Load data
    print(f"\n📂 Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"   Loaded {len(df)} rows")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Prepare data
    X = df[['Pitch', 'Time']]
    port_names = ['TDRDF1', 'TDRDF2', 'TDRDF3', 'TDRDF4']
    
    # Train models
    fine_tuned_model = FineTunedTDRModel()
    
    for port in port_names:
        if port in df.columns:
            y = df[port]
            fine_tuned_model.train_tdr_model(X, y, port)
        else:
            print(f"\n⚠️ Warning: Column '{port}' not found in CSV!")
    
    # Save model
    print("\n" + "="*60)
    print("Saving model...")
    print("="*60)
    joblib.dump(fine_tuned_model, 'fine_tuned_tdr_model.pkl')
    print("✅ Model saved as 'fine_tuned_tdr_model.pkl'")
    
    # Test predictions
    print("\n" + "="*60)
    print("Sample Predictions")
    print("="*60)
    
    test_cases = [
        (3.3, 0.0),
        (3.3, 0.5),
        (3.3, 1.0)
    ]
    
    for pitch, time in test_cases:
        print(f"\nPitch={pitch}, Time={time}ns:")
        for port in port_names:
            if port in df.columns:
                pred = fine_tuned_model.predict(port, pitch, time)
                print(f"  {port}: {pred:.2f} Ω")
    
    print("\n✅ Training complete! You can now run comparison_script.py")

if __name__ == "__main__":
    main()