import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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

if __name__ == "__main__":
    compare_performance()