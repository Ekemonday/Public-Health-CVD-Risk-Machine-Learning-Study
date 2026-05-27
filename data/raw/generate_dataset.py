import pandas as pd
import numpy as np

np.random.seed(42)
N = 2000

age = np.random.normal(52, 15, N).clip(18, 90).astype(int)
gender = np.random.choice(['Male', 'Female'], N, p=[0.49, 0.51])
region = np.random.choice(['North', 'South', 'East', 'West', 'Central'], N)
urban_rural = np.random.choice(['Urban', 'Rural', 'Suburban'], N, p=[0.5, 0.25, 0.25])
education = np.random.choice(['None', 'Primary', 'Secondary', 'Tertiary'], N, p=[0.1, 0.25, 0.40, 0.25])
income_level = np.random.choice(['Low', 'Middle', 'High'], N, p=[0.35, 0.45, 0.20])
marital_status = np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], N, p=[0.25, 0.55, 0.12, 0.08])

bmi = np.random.normal(27.5, 5.5, N).clip(15, 55)
systolic_bp = (110 + 0.4 * age + 5 * (gender == 'Male') + np.random.normal(0, 12, N)).clip(80, 200)
diastolic_bp = (70 + 0.2 * age + 3 * (gender == 'Male') + np.random.normal(0, 8, N)).clip(50, 130)
heart_rate = np.random.normal(75, 12, N).clip(45, 130).astype(int)
waist_cm = (75 + 0.3 * age + 10 * (gender == 'Male') + 1.5 * bmi + np.random.normal(0, 8, N)).clip(55, 160)

blood_glucose = (85 + 0.15 * age + 0.8 * bmi + np.random.normal(0, 15, N)).clip(60, 400)
hba1c = (4.5 + 0.01 * blood_glucose + np.random.normal(0, 0.5, N)).clip(4, 14)
total_chol = (160 + 0.5 * age + 0.7 * bmi + np.random.normal(0, 25, N)).clip(100, 380)
ldl = (total_chol * 0.6 + np.random.normal(0, 15, N)).clip(50, 260)
hdl = (65 - 0.2 * age - 5 * (gender == 'Male') + np.random.normal(0, 10, N)).clip(20, 100)
trig = (100 + 0.5 * age + 1.2 * bmi + np.random.normal(0, 40, N)).clip(50, 600)
creatinine = np.random.normal(0.95, 0.25, N).clip(0.4, 4.0)

smoking = np.random.choice(['Never', 'Former', 'Current'], N, p=[0.55, 0.25, 0.20])
alcohol = np.random.choice(['None', 'Moderate', 'Heavy'], N, p=[0.40, 0.45, 0.15])
activity = np.random.choice(['Sedentary', 'Low', 'Moderate', 'High'], N, p=[0.30, 0.30, 0.25, 0.15])
diet = np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], N, p=[0.25, 0.35, 0.28, 0.12])
sleep = np.random.normal(6.8, 1.2, N).clip(3, 12).round(1)
stress = np.random.randint(1, 11, N)

hypertension = ((systolic_bp > 140) | (diastolic_bp > 90)).astype(int)
diabetes = (blood_glucose > 126).astype(int)
fam_cvd = np.random.choice([0,1], N, p=[0.60,0.40])
fam_dm = np.random.choice([0,1], N, p=[0.65,0.35])
prev_mi = np.random.choice([0,1], N, p=[0.92,0.08])
stroke_hx = np.random.choice([0,1], N, p=[0.95,0.05])
kidney_dis = np.random.choice([0,1], N, p=[0.93,0.07])

insurance = np.random.choice(['None','Partial','Full'], N, p=[0.25,0.35,0.40])
dist_clinic = np.random.exponential(8, N).clip(0.5, 80).round(1)
hosp_visits = np.random.poisson(2.5, N)
med_adhere = np.random.choice(['Poor','Moderate','Good'], N, p=[0.20,0.35,0.45])

# CVD Risk Score (regression) — realistic range 1-40
cvd_risk = (
    0.08 * (age - 18) +
    3.0 * (gender == 'Male').astype(float) +
    0.06 * (systolic_bp - 80) +
    0.03 * (ldl - 50) +
    -0.05 * (hdl - 20) +
    0.02 * (blood_glucose - 60) +
    3.5 * (smoking == 'Current').astype(float) +
    1.5 * (smoking == 'Former').astype(float) +
    0.2 * (bmi - 15) +
    3.0 * fam_cvd +
    5.0 * prev_mi +
    2.5 * diabetes +
    1.5 * (activity == 'Sedentary').astype(float) +
    0.3 * stress +
    np.random.normal(0, 1.5, N)
)
# Normalize to realistic 1–40 range
cvd_risk = ((cvd_risk - cvd_risk.min()) / (cvd_risk.max() - cvd_risk.min()) * 39 + 1).clip(1, 40)

# Hospitalization risk (classification) — balanced ~35% positive
hosp_log_odds = (
    -3.5 +
    0.04 * age +
    0.012 * systolic_bp +
    0.008 * blood_glucose +
    1.8 * prev_mi +
    1.5 * stroke_hx +
    1.2 * kidney_dis +
    0.8 * (smoking == 'Current').astype(float) +
    0.6 * (activity == 'Sedentary').astype(float) +
    0.5 * (med_adhere == 'Poor').astype(float) +
    0.4 * hypertension +
    np.random.normal(0, 0.5, N)
)
hosp_prob = 1 / (1 + np.exp(-hosp_log_odds))
hospitalized = (hosp_prob > 0.5).astype(int)

df = pd.DataFrame({
    'patient_id': [f'PH{str(i).zfill(5)}' for i in range(1,N+1)],
    'age': age, 'gender': gender, 'region': region,
    'urban_rural': urban_rural, 'education_level': education,
    'income_level': income_level, 'marital_status': marital_status,
    'bmi': bmi.round(1), 'systolic_bp': systolic_bp.round(0).astype(int),
    'diastolic_bp': diastolic_bp.round(0).astype(int), 'heart_rate': heart_rate,
    'waist_circumference_cm': waist_cm.round(1),
    'blood_glucose_mgdl': blood_glucose.round(1),
    'hba1c_percent': hba1c.round(2),
    'total_cholesterol_mgdl': total_chol.round(1),
    'ldl_cholesterol_mgdl': ldl.round(1),
    'hdl_cholesterol_mgdl': hdl.round(1),
    'triglycerides_mgdl': trig.round(1),
    'creatinine_mgdl': creatinine.round(2),
    'smoking_status': smoking, 'alcohol_consumption': alcohol,
    'physical_activity_level': activity, 'diet_quality': diet,
    'sleep_hours_per_night': sleep, 'stress_level': stress,
    'hypertension': hypertension, 'diabetes': diabetes,
    'family_history_cvd': fam_cvd, 'family_history_diabetes': fam_dm,
    'previous_heart_attack': prev_mi, 'stroke_history': stroke_hx,
    'kidney_disease': kidney_dis, 'insurance_coverage': insurance,
    'distance_to_clinic_km': dist_clinic,
    'hospital_visits_per_year': hosp_visits,
    'medication_adherence': med_adhere,
    'cvd_risk_score': cvd_risk.round(2),
    'hospitalized_30days': hospitalized
})

# 3% missing in lab values (realistic)
for col in ['hba1c_percent','ldl_cholesterol_mgdl','triglycerides_mgdl','creatinine_mgdl']:
    idx = np.random.choice(df.index, size=int(0.03*N), replace=False)
    df.loc[idx, col] = np.nan

df.to_csv('/home/claude/ml_public_health/data/raw/public_health_dataset.csv', index=False)
print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} cols")
print("CVD Risk:", df['cvd_risk_score'].describe().round(2).to_dict())
print("Hospitalized:", df['hospitalized_30days'].value_counts().to_dict())
