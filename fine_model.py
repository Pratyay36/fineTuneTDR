import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

class FineTunedTDRModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.imputers = {}
        self.best_params = {}
        
    def create_features_safe(self, X):
        """Create advanced features WITHOUT generating NaN values"""
        X_new = X.copy()
        
        # Original features
        pitch = X['Pitch'].values
        time = X['Time'].values
        
        # Ensure no negative or zero values for safe operations
        time_safe = np.maximum(time, 1e-6)  # Replace very small values with 1e-6
        
        # Polynomial features (safe - no NaN)
        X_new['Time_squared'] = time ** 2
        X_new['Time_cubed'] = time ** 3
        
        # Square root (safe with clipping)
        X_new['Time_sqrt'] = np.sqrt(np.maximum(time, 0))
        
        # Interaction features
        X_new['Pitch_Time'] = pitch * time
        X_new['Pitch_Time_squared'] = pitch * (time ** 2)
        
        # Exponential features (always safe)
        X_new['Time_exp'] = np.exp(-time)
        X_new['Time_exp_squared'] = np.exp(-(time ** 2))
        
        # Logarithmic features (safe with clipping)
        X_new['Time_log'] = np.log(np.maximum(time + 0.001, 0.001))
        
        # Sinusoidal features (safe)
        X_new['Time_sin'] = np.sin(2 * np.pi * time)
        X_new['Time_cos'] = np.cos(2 * np.pi * time)
        
        # Replace any potential NaN with 0
        X_new = X_new.fillna(0)
        X_new = X_new.replace([np.inf, -np.inf], 0)
        
        return X_new
    
    def train_with_cross_validation(self, X_train, y_train, model_name, base_model, param_grid):
        """Train model with cross-validation and hyperparameter tuning"""
        
        print(f"\n🔧 Fine-tuning {model_name}...")
        
        # Ensure no NaN in training data
        if np.any(np.isnan(X_train)):
            print(f"  ⚠️ Found NaN in {model_name} training data, applying imputation...")
            imputer = SimpleImputer(strategy='median')
            X_train = imputer.fit_transform(X_train)
        else:
            imputer = None
        
        # Use RandomizedSearchCV for faster tuning
        random_search = RandomizedSearchCV(
            base_model, param_grid, 
            n_iter=10, cv=3,  # Reduced iterations for speed
            scoring='r2', 
            n_jobs=-1, 
            random_state=42,
            verbose=0
        )
        
        random_search.fit(X_train, y_train)
        
        best_model = random_search.best_estimator_
        best_score = random_search.best_score_
        best_params = random_search.best_params_
        
        print(f"  ✓ Best R² (CV): {best_score:.4f}")
        print(f"  ✓ Best params: {best_params}")
        
        return best_model, best_score, best_params, imputer
    
    def train_ensemble_model(self, X_train, y_train, X_val, y_val):
        """Train multiple models and ensemble them"""
        
        # Ensure no NaN in validation data
        if np.any(np.isnan(X_val)):
            X_val = np.nan_to_num(X_val, nan=0.0)
        
        models_info = {}
        
        # 1. Random Forest with fine-tuning
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf_model, rf_score, rf_params_best, rf_imputer = self.train_with_cross_validation(
            X_train, y_train, "Random Forest",
            RandomForestRegressor(random_state=42, n_jobs=-1),
            rf_params
        )
        models_info['RandomForest'] = {
            'model': rf_model, 
            'score': rf_score, 
            'weight': max(rf_score, 0.1),
            'imputer': rf_imputer
        }
        
        # 2. Gradient Boosting with fine-tuning (using HistGradientBoosting to avoid NaN issues)
        from sklearn.ensemble import HistGradientBoostingRegressor
        
        gb_params = {
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [5, 7, 10],
            'min_samples_leaf': [10, 20, 50],
            'l2_regularization': [0.0, 0.1, 1.0]
        }
        
        print(f"\n🔧 Fine-tuning Gradient Boosting...")
        gb_base = HistGradientBoostingRegressor(random_state=42, max_iter=100)
        
        random_search_gb = RandomizedSearchCV(
            gb_base, gb_params, 
            n_iter=10, cv=3, 
            scoring='r2', 
            n_jobs=-1, 
            random_state=42,
            verbose=0
        )
        
        random_search_gb.fit(X_train, y_train)
        gb_model = random_search_gb.best_estimator_
        gb_score = random_search_gb.best_score_
        
        print(f"  ✓ Best R² (CV): {gb_score:.4f}")
        models_info['GradientBoosting'] = {
            'model': gb_model, 
            'score': gb_score, 
            'weight': max(gb_score, 0.1),
            'imputer': None
        }
        
        # 3. Neural Network with fine-tuning
        nn_params = {
            'hidden_layer_sizes': [(50,), (100,), (50, 25)],
            'activation': ['relu'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        }
        nn_model, nn_score, nn_params_best, nn_imputer = self.train_with_cross_validation(
            X_train, y_train, "Neural Network",
            MLPRegressor(max_iter=1000, random_state=42, verbose=False, early_stopping=True),
            nn_params
        )
        models_info['NeuralNetwork'] = {
            'model': nn_model, 
            'score': nn_score, 
            'weight': max(nn_score, 0.1),
            'imputer': nn_imputer
        }
        
        # 4. Ridge Regression with fine-tuning
        ridge_params = {
            'alpha': [0.01, 0.1, 1.0, 10.0]
        }
        ridge_model, ridge_score, ridge_params_best, ridge_imputer = self.train_with_cross_validation(
            X_train, y_train, "Ridge Regression",
            Ridge(random_state=42),
            ridge_params
        )
        models_info['Ridge'] = {
            'model': ridge_model, 
            'score': ridge_score, 
            'weight': max(ridge_score, 0.1),
            'imputer': ridge_imputer
        }
        
        # Calculate normalized weights
        total_weight = sum(info['weight'] for info in models_info.values())
        for info in models_info.values():
            info['weight'] = info['weight'] / total_weight
        
        return models_info
    
    def ensemble_predict(self, models_info, X):
        """Weighted average prediction from ensemble"""
        predictions = []
        weights = []
        
        for model_info in models_info.values():
            X_input = X.copy()
            if model_info['imputer'] is not None:
                X_input = model_info['imputer'].transform(X_input)
            pred = model_info['model'].predict(X_input)
            predictions.append(pred)
            weights.append(model_info['weight'])
        
        # Weighted average
        weights = np.array(weights).reshape(-1, 1)
        predictions = np.array(predictions)
        weighted_pred = np.average(predictions, axis=0, weights=weights.flatten())
        
        return weighted_pred
    
    def train_tdr_model(self, X, y, port_name, use_ensemble=True):
        """Complete training pipeline for TDR prediction"""
        
        print(f"\n{'='*60}")
        print(f"Fine-Tuning Model for {port_name}")
        print(f"{'='*60}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create advanced features
        print("\n🔧 Creating advanced features...")
        X_train_feat = self.create_features_safe(X_train)
        X_val_feat = self.create_features_safe(X_val)
        
        print(f"  Original features: {X_train.shape[1]}")
        print(f"  Engineered features: {X_train_feat.shape[1]}")
        
        # Apply robust scaling
        print("\n📊 Applying robust scaling...")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_feat)
        X_val_scaled = scaler.transform(X_val_feat)
        
        # Final check for NaN
        if np.any(np.isnan(X_train_scaled)):
            print("  ⚠️ Replacing remaining NaN values...")
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
        if np.any(np.isnan(X_val_scaled)):
            X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0)
        
        if use_ensemble:
            # Train ensemble of models
            print("\n🎯 Training ensemble model...")
            models_info = self.train_ensemble_model(X_train_scaled, y_train, X_val_scaled, y_val)
            
            # Make predictions
            y_pred = self.ensemble_predict(models_info, X_val_scaled)
            
            # Evaluate
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            print(f"\n📈 Ensemble Performance on Validation Set:")
            print(f"  R² Score: {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f} Ω")
            
            # Save model
            self.models[port_name] = {
                'type': 'ensemble',
                'models': models_info,
                'scaler': scaler,
                'feature_engineer': self.create_features_safe
            }
            
        else:
            # Train single best model
            print("\n🎯 Training single best model...")
            best_model, best_score, best_params, best_imputer = self.train_with_cross_validation(
                X_train_scaled, y_train, "Best Model",
                RandomForestRegressor(random_state=42, n_jobs=-1),
                {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [2, 5]
                }
            )
            
            # Validate
            y_pred = best_model.predict(X_val_scaled)
            r2 = r2_score(y_val, y_pred)
            
            print(f"\n📈 Model Performance on Validation Set:")
            print(f"  R² Score: {r2:.4f}")
            
            self.models[port_name] = {
                'type': 'single',
                'model': best_model,
                'scaler': scaler,
                'feature_engineer': self.create_features_safe,
                'imputer': best_imputer
            }
        
        return self.models[port_name]
    
    def predict(self, port_name, pitch, time):
        """Make predictions using fine-tuned model"""
        
        if port_name not in self.models:
            raise ValueError(f"Model for {port_name} not found. Train first!")
        
        model_info = self.models[port_name]
        
        # Create input dataframe
        X = pd.DataFrame({'Pitch': [pitch], 'Time': [time]})
        
        # Apply feature engineering
        X_feat = model_info['feature_engineer'](X)
        
        # Scale features
        X_scaled = model_info['scaler'].transform(X_feat)
        
        # Apply imputer if exists
        if 'imputer' in model_info and model_info['imputer'] is not None:
            X_scaled = model_info['imputer'].transform(X_scaled)
        
        # Predict
        if model_info['type'] == 'ensemble':
            prediction = self.ensemble_predict(model_info['models'], X_scaled)
        else:
            prediction = model_info['model'].predict(X_scaled)
        
        return prediction[0]

def load_and_prepare_data(csv_file_path):
    """Load and prepare TDR data with outlier handling"""
    
    df = pd.read_csv(csv_file_path)
    
    print("\n📊 Data Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Time range: {df['Time'].min()} to {df['Time'].max()}")
    
    # Check for NaN in original data
    if df.isnull().any().any():
        print("  ⚠️ Found NaN values, filling with median...")
        df = df.fillna(df.median())
    
    # Handle outliers using IQR method
    for col in ['TDRDF1', 'TDRDF2', 'TDRDF3', 'TDRDF4']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing them
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    print(f"  Data ready for training")
    
    return df

def main():
    print("="*60)
    print("FINE-TUNED TDR PREDICTION MODEL (FIXED VERSION)")
    print("="*60)
    
    # Load data
    csv_file_path = 'tdr_data.csv'
    
    if not os.path.exists(csv_file_path):
        print(f"❌ Error: CSV file '{csv_file_path}' not found!")
        print("Make sure the file is in the current directory")
        return
    
    df = load_and_prepare_data(csv_file_path)
    
    # Prepare features and targets
    X = df[['Pitch', 'Time']]
    
    port_names = ['TDRDF1', 'TDRDF2', 'TDRDF3', 'TDRDF4']
    
    # Initialize fine-tuned model
    fine_tuned_model = FineTunedTDRModel()
    
    # Train models for each port
    for port in port_names:
        y = df[port]
        fine_tuned_model.train_tdr_model(X, y, port, use_ensemble=True)
    
    # Save all models
    print("\n💾 Saving fine-tuned models...")
    joblib.dump(fine_tuned_model, 'fine_tuned_tdr_model.pkl')
    print("✓ Saved fine_tuned_tdr_model.pkl")
    
    # Test predictions
    print("\n" + "="*60)
    print("TESTING PREDICTIONS")
    print("="*60)
    
    test_cases = [
        (3.3, 0.0),
        (3.3, 0.5),
        (3.3, 1.0),
        (3.3, 0.25),
        (3.3, 0.75),
    ]
    
    for pitch, time in test_cases:
        print(f"\n📊 Pitch={pitch}, Time={time}ns:")
        for port in port_names:
            try:
                pred = fine_tuned_model.predict(port, pitch, time)
                print(f"  {port}: {pred:.4f} Ω")
            except Exception as e:
                print(f"  {port}: Error - {e}")
    
    # Generate full waveform prediction
    print("\n" + "="*60)
    print("GENERATING FULL TDR WAVEFORM")
    print("="*60)
    
    pitch_fixed = 3.3
    times = np.linspace(0, 1.1, 30)  # Reduced points for speed
    
    waveform_data = {'Time': times}
    for port in port_names:
        predictions = []
        for t in times:
            try:
                pred = fine_tuned_model.predict(port, pitch_fixed, t)
                predictions.append(pred)
            except:
                predictions.append(np.nan)
        waveform_data[port] = predictions
    
    waveform_df = pd.DataFrame(waveform_data)
    waveform_df.to_csv('fine_tuned_tdr_waveform.csv', index=False)
    print(f"✓ Saved fine_tuned_tdr_waveform.csv")
    
    print("\n✅ Fine-tuning complete successfully!")
    print("\n📈 Model Features:")
    print("  • Safe feature engineering (no NaN values)")
    print("  • Ensemble learning with 4 algorithms")
    print("  • Robust scaling for outliers")
    print("  • NaN handling with imputation")
    print("  • HistGradientBoosting (handles NaN natively)")

def quick_predict(pitch, time):
    """Quick prediction using saved fine-tuned model"""
    try:
        model = joblib.load('fine_tuned_tdr_model.pkl')
        port_names = ['TDRDF1', 'TDRDF2', 'TDRDF3', 'TDRDF4']
        
        predictions = {}
        for port in port_names:
            predictions[port] = model.predict(port, pitch, time)
        
        return predictions
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    main()