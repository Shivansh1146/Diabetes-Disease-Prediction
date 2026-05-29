"""
Download the original Pima Indians Diabetes Dataset CSV file.
"""
import pandas as pd
import os

def download_dataset():
    # Reliable raw URL for the original Pima Indians Diabetes Dataset
    url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
    print(f"Downloading original dataset from {url}...")
    try:
        df = pd.read_csv(url)
        
        # Ensure columns match what the model expects
        expected_cols = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ]
        
        # If the downloaded dataset doesn't have headers, or has different names, we enforce them.
        # This specific URL actually has the exact headers already, but we'll enforce them just in case.
        if len(df.columns) == 9:
            df.columns = expected_cols
            
        output_path = 'diabetes.csv'
        df.to_csv(output_path, index=False)
        print(f"[OK] Original dataset successfully downloaded and saved to {output_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Diabetic cases: {df['Outcome'].sum()} ({(df['Outcome'].sum()/len(df))*100:.1f}%)")
        
    except Exception as e:
        print(f"[Error] Error downloading dataset: {e}")

if __name__ == '__main__':
    download_dataset()
