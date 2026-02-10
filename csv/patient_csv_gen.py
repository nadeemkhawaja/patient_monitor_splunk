import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Generating patient data...")

# Patient 1 - Sepsis
data1 = []
base_time = datetime.now()
for i in range(60):
    data1.append({
        'patient_id': 'P001',
        'timestamp': (base_time + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S"),
        'ECG': 'Sinus Tachycardia',
        'heart_rate_bpm': np.random.randint(90, 150),
        'temperature_c': round(np.random.uniform(37.5, 40.0), 1),
        'bp_systolic_mmHg': np.random.randint(80, 110),
        'bp_diastolic_mmHg': np.random.randint(50, 70),
        'spo2_percent': np.random.randint(88, 96)
    })

df1 = pd.DataFrame(data1)
df1.to_csv('patient_1_sepsis.csv', index=False)
print(f"✅ Created patient_1_sepsis.csv")

# Patient 2 - V-Tach
data2 = []
for i in range(60):
    if 38 <= i <= 42:
        hr = np.random.randint(160, 180)
        ecg = 'V-Tach'
        spo2 = np.random.randint(85, 92)
    else:
        hr = np.random.randint(70, 90)
        ecg = 'Normal Sinus Rhythm'
        spo2 = np.random.randint(95, 99)
    
    data2.append({
        'patient_id': 'P002',
        'timestamp': (base_time + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S"),
        'ECG': ecg,
        'heart_rate_bpm': hr,
        'temperature_c': round(np.random.uniform(36.8, 37.5), 1),
        'bp_systolic_mmHg': np.random.randint(110, 130),
        'bp_diastolic_mmHg': np.random.randint(65, 85),
        'spo2_percent': spo2
    })

df2 = pd.DataFrame(data2)
df2.to_csv('patient_2_arrhythmia.csv', index=False)
print(f"✅ Created patient_2_arrhythmia.csv")

# Patient 3 - Respiratory
data3 = []
for i in range(60):
    data3.append({
        'patient_id': 'P003',
        'timestamp': (base_time + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S"),
        'ECG': 'Sinus Tachycardia',
        'heart_rate_bpm': np.random.randint(95, 120),
        'temperature_c': round(np.random.uniform(37.0, 38.0), 1),
        'bp_systolic_mmHg': np.random.randint(105, 125),
        'bp_diastolic_mmHg': np.random.randint(60, 80),
        'spo2_percent': np.random.randint(85, 93)
    })

df3 = pd.DataFrame(data3)
df3.to_csv('patient_3_respiratory.csv', index=False)
print(f"✅ Created patient_3_respiratory.csv")

print("\n✅ All files created successfully!")
print("Run: streamlit run patient_monitor.py")
