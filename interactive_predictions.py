import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Class definition (must match the trained model)
class TDRAnalyzer:
    def __init__(self):
        self.ports = ['TDRDF1', 'TDRDF2', 'TDRDF3', 'TDRDF4']
        self.models = {}
        self.scalers = {}
    
    def create_features(self, pitch, time):
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
    
    def predict(self, port_name, pitch, time):
        if port_name not in self.models:
            raise ValueError(f"Port {port_name} not trained")
        
        X = self.create_features(pitch, time)
        X_scaled = self.scalers[port_name].transform(X)
        
        rf_pred = self.models[port_name]['rf'].predict(X_scaled)[0]
        gb_pred = self.models[port_name]['gb'].predict(X_scaled)[0]
        ridge_pred = self.models[port_name]['ridge'].predict(X_scaled)[0]
        
        return (rf_pred + gb_pred + ridge_pred) / 3

# Load the model
print("="*60)
print("LOADING TDR PREDICTION MODEL")
print("="*60)

try:
    model = joblib.load('tdr_analyzer.pkl')
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Model file 'tdr_analyzer.pkl' not found!")
    print("Please run complete_tdr_solution.py first to train the model.")
    exit()

print("\n" + "="*60)
print("INTERACTIVE TDR PREDICTION TOOL")
print("="*60)
print("\n📊 This tool predicts TDR values for 4 ports:")
print("   - TDRDF1, TDRDF2, TDRDF3, TDRDF4")
print("\n💡 Enter pitch and time values to get predictions")
print("   Type 'quit' to exit")
print("   Type 'batch' to predict multiple values")
print("   Type 'waveform' to generate a full waveform")
print("="*60)

def predict_single():
    """Single prediction mode"""
    while True:
        print("\n" + "-"*40)
        try:
            pitch_input = input("Enter Pitch (e.g., 3.3): ")
            if pitch_input.lower() == 'quit':
                return False
            if pitch_input.lower() == 'menu':
                return True
            
            pitch = float(pitch_input)
            time = float(input("Enter Time in ns (e.g., 0.5): "))
            
            print(f"\n📊 PREDICTIONS for Pitch={pitch}, Time={time}ns")
            print("="*50)
            
            for port in model.ports:
                pred = model.predict(port, pitch, time)
                print(f"  {port}: {pred:.2f} Ω")
            
            print("="*50)
            
        except ValueError:
            print("❌ Invalid input! Please enter numbers.")
        except Exception as e:
            print(f"❌ Error: {e}")
    return True

def batch_predict():
    """Batch prediction mode"""
    print("\n" + "="*60)
    print("BATCH PREDICTION MODE")
    print("="*60)
    print("Enter multiple pitch,time pairs")
    print("Type 'done' when finished")
    
    predictions = []
    
    while True:
        try:
            user_input = input("\nEnter pitch,time (e.g., 3.3,0.5): ")
            if user_input.lower() == 'done':
                break
            if user_input.lower() == 'menu':
                return
            
            pitch, time = map(float, user_input.split(','))
            
            row = {'pitch': pitch, 'time_ns': time}
            for port in model.ports:
                row[f'{port}_ohms'] = model.predict(port, pitch, time)
            predictions.append(row)
            
            print(f"  ✅ Added: Pitch={pitch}, Time={time}ns")
            
        except ValueError:
            print("  ❌ Invalid format! Use: pitch,time (e.g., 3.3,0.5)")
    
    if predictions:
        df = pd.DataFrame(predictions)
        output_file = 'batch_predictions_output.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✅ Saved {len(predictions)} predictions to '{output_file}'")
        print("\n📊 Results:")
        print(df.round(2).to_string(index=False))

def generate_waveform():
    """Generate full TDR waveform"""
    print("\n" + "="*60)
    print("TDR WAVEFORM GENERATION")
    print("="*60)
    
    try:
        pitch = float(input("Enter Pitch value (e.g., 3.3): "))
        time_start = float(input("Enter Start Time (ns, default 0): ") or 0)
        time_end = float(input("Enter End Time (ns, default 1.1): ") or 1.1)
        num_points = int(input("Enter Number of points (default 50): ") or 50)
        
        print(f"\n📊 Generating waveform for Pitch={pitch}...")
        
        times = np.linspace(time_start, time_end, num_points)
        waveform_data = {'Time_ns': times}
        
        for port in model.ports:
            predictions = [model.predict(port, pitch, t) for t in times]
            waveform_data[port] = predictions
        
        df = pd.DataFrame(waveform_data)
        output_file = f'tdr_waveform_pitch_{pitch}.csv'
        df.to_csv(output_file, index=False)
        
        print(f"\n✅ Waveform saved to '{output_file}'")
        print(f"   Generated {num_points} points from {time_start} to {time_end} ns")
        print("\n📊 First 10 rows:")
        print(df.head(10).round(2))
        
    except ValueError as e:
        print(f"❌ Invalid input: {e}")

def main():
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. 🔮 Single Prediction")
        print("2. 📊 Batch Predictions")
        print("3. 📈 Generate TDR Waveform")
        print("4. ❌ Exit")
        print("="*60)
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            predict_single()
        elif choice == '2':
            batch_predict()
        elif choice == '3':
            generate_waveform()
        elif choice == '4':
            print("\n👋 Thank you for using TDR Prediction Tool!")
            break
        else:
            print("❌ Invalid choice! Please enter 1, 2, 3, or 4")

if __name__ == "__main__":
    main()