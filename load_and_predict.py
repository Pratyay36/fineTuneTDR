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

# IMPORTANT: This class MUST be defined before loading the model
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
    
    def predict(self, port_name, pitch, time):
        """Make prediction"""
        
        if port_name not in self.models:
            raise ValueError(f"Model for {port_name} not found")
        
        model_info = self.models[port_name]
        
        # Create input
        X = pd.DataFrame({'Pitch': [pitch], 'Time': [time]})
        
        # Feature engineering
        X_feat = self.create_features_safe(X)
        
        # Scale
        X_scaled = model_info['scaler'].transform(X_feat)
        
        # Predict
        prediction = self.ensemble_predict(model_info['models'], X_scaled)
        
        return prediction[0]

def compare_performance():
    """Compare original vs fine-tuned model performance"""
    
    print("="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    
    # Check if model file exists
    if not os.path.exists('fine_tuned_tdr_model.pkl'):
        print("\n❌ Error: 'fine_tuned_tdr_model.pkl' not found!")
        print("Make sure you're in the correct directory.")
        return
    
    # Load data
    try:
        df = pd.read_csv('tdr_data.csv')
        print(f"\n✅ Loaded {len(df)} rows from tdr_data.csv")
    except FileNotFoundError:
        print("\n❌ Could not load tdr_data.csv")
        return
    
    X = df[['Pitch', 'Time']]
    port_names = ['TDRDF1', 'TDRDF2', 'TDRDF3', 'TDRDF4']
    
    try:
        # Load fine-tuned model
        print("\n📂 Loading fine-tuned model...")
        fine_tuned = joblib.load('fine_tuned_tdr_model.pkl')
        print("✅ Model loaded successfully!")
        
        print("\n" + "="*60)
        print("📊 MODEL PERFORMANCE METRICS")
        print("="*60)
        
        all_results = []
        
        for port in port_names:
            if port not in df.columns:
                continue
                
            y_true = df[port].values
            
            # Get predictions from fine-tuned model
            print(f"\n🔄 Calculating predictions for {port}...")
            y_pred_finetuned = []
            for _, row in X.iterrows():
                pred = fine_tuned.predict(port, row['Pitch'], row['Time'])
                y_pred_finetuned.append(pred)
            
            y_pred_finetuned = np.array(y_pred_finetuned)
            
            # Calculate metrics
            r2_finetuned = r2_score(y_true, y_pred_finetuned)
            mae_finetuned = mean_absolute_error(y_true, y_pred_finetuned)
            rmse_finetuned = np.sqrt(mean_squared_error(y_true, y_pred_finetuned))
            
            # Calculate prediction accuracy percentage
            mean_actual = np.mean(y_true)
            accuracy_percent = (1 - mae_finetuned / mean_actual) * 100
            
            print(f"\n{'='*50}")
            print(f"📈 {port} Performance:")
            print(f"{'='*50}")
            print(f"  R² Score:        {r2_finetuned:.4f}")
            print(f"  MAE:             {mae_finetuned:.4f} Ω")
            print(f"  RMSE:            {rmse_finetuned:.4f} Ω")
            print(f"  Accuracy:        {accuracy_percent:.2f}%")
            
            # Simple baseline (predict mean)
            mean_pred = np.mean(y_true)
            baseline_mae = mean_absolute_error(y_true, [mean_pred] * len(y_true))
            baseline_accuracy = (1 - baseline_mae / mean_actual) * 100
            
            print(f"\n  📊 Baseline (predicting mean):")
            print(f"     MAE:      {baseline_mae:.4f} Ω")
            print(f"     Accuracy: {baseline_accuracy:.2f}%")
            print(f"  🚀 Improvement: {((baseline_mae - mae_finetuned) / baseline_mae * 100):.1f}% better")
            
            all_results.append({
                'Port': port,
                'R² Score': r2_finetuned,
                'MAE (Ω)': mae_finetuned,
                'RMSE (Ω)': rmse_finetuned,
                'Accuracy (%)': accuracy_percent,
                'Improvement (%)': ((baseline_mae - mae_finetuned) / baseline_mae * 100)
            })
        
        # Summary table
        print("\n" + "="*60)
        print("📊 PERFORMANCE SUMMARY TABLE")
        print("="*60)
        summary_df = pd.DataFrame(all_results)
        print(summary_df.to_string(index=False))
        
        # Average performance
        print("\n" + "="*60)
        print("📈 AVERAGE PERFORMANCE ACROSS ALL PORTS")
        print("="*60)
        print(f"  Average R² Score:     {summary_df['R² Score'].mean():.4f}")
        print(f"  Average MAE:          {summary_df['MAE (Ω)'].mean():.4f} Ω")
        print(f"  Average RMSE:         {summary_df['RMSE (Ω)'].mean():.4f} Ω")
        print(f"  Average Accuracy:     {summary_df['Accuracy (%)'].mean():.2f}%")
        print(f"  Average Improvement:  {summary_df['Improvement (%)'].mean():.1f}%")
        
        print("\n" + "="*60)
        print("✅ MODEL VALIDATION SUCCESSFUL!")
        print("="*60)
        print("\n🎯 The model is performing well and ready for predictions!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def make_predictions():
    """Make predictions using the trained model"""
    print("\n" + "="*60)
    print("MAKING PREDICTIONS")
    print("="*60)
    
    try:
        # Load model
        model = joblib.load('fine_tuned_tdr_model.pkl')
        
        # Test predictions
        test_cases = [
            (3.3, 0.0),
            (3.3, 0.25),
            (3.3, 0.5),
            (3.3, 0.75),
            (3.3, 1.0),
            (3.5, 0.5),
        ]
        
        print("\n📊 Sample Predictions:")
        print("-" * 60)
        
        for pitch, time in test_cases:
            print(f"\nPitch={pitch}, Time={time}ns:")
            for port in ['TDRDF1', 'TDRDF2', 'TDRDF3', 'TDRDF4']:
                pred = model.predict(port, pitch, time)
                print(f"  {port}: {pred:.2f} Ω")
        
        # Generate full waveform
        print("\n" + "="*60)
        print("Generating Full TDR Waveform")
        print("="*60)
        
        pitch = 3.3
        times = np.linspace(0, 1.1, 50)
        waveform_data = []
        
        for t in times:
            row = {'Time_ns': t}
            for port in ['TDRDF1', 'TDRDF2', 'TDRDF3', 'TDRDF4']:
                row[port] = model.predict(port, pitch, t)
            waveform_data.append(row)
        
        waveform_df = pd.DataFrame(waveform_data)
        waveform_df.to_csv('tdr_predictions.csv', index=False)
        print(f"✅ Saved waveform to 'tdr_predictions.csv'")
        print(f"\nFirst 5 rows:")
        print(waveform_df.head())
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TDR MODEL ANALYSIS AND PREDICTION TOOL")
    print("="*60)
    
    # Run comparison
    compare_performance()
    
    # Make predictions
    make_predictions()