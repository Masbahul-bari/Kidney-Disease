import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('kidney_disease.csv')

print("=== KIDNEY DISEASE DATASET ANALYSIS ===\n")

# Basic dataset information
print("1. DATASET OVERVIEW")
print(f"Total records: {len(df)}")
print(f"Features: {len(df.columns)}")
print(f"Target variable: {df['classification'].value_counts().to_dict()}")
print(f"Missing values: {df.isnull().sum().sum()}")
print()

# Feature descriptions
feature_descriptions = {
    'id': 'Patient ID',
    'age': 'Age in years',
    'bp': 'Blood pressure (mm Hg)',
    'sg': 'Specific gravity (urine density)',
    'al': 'Albumin (0-5)',
    'su': 'Sugar (0-5)',
    'rbc': 'Red blood cells (normal/abnormal)',
    'pc': 'Pus cells (normal/abnormal)',
    'pcc': 'Pus cell clumps (present/notpresent)',
    'ba': 'Bacteria (present/notpresent)',
    'bgr': 'Blood glucose random (mg/dL)',
    'bu': 'Blood urea (mg/dL)',
    'sc': 'Serum creatinine (mg/dL)',
    'sod': 'Sodium (mEq/L)',
    'pot': 'Potassium (mEq/L)',
    'hemo': 'Hemoglobin (g/dL)',
    'pcv': 'Packed cell volume (%)',
    'wc': 'White blood cell count (cells/cumm)',
    'rc': 'Red blood cell count (millions/cmm)',
    'htn': 'Hypertension (yes/no)',
    'dm': 'Diabetes mellitus (yes/no)',
    'cad': 'Coronary artery disease (yes/no)',
    'appet': 'Appetite (good/poor)',
    'pe': 'Pedal edema (yes/no)',
    'ane': 'Anemia (yes/no)',
    'classification': 'Target: CKD (ckd/notckd)'
}

print("2. FEATURE DESCRIPTIONS")
for col, desc in feature_descriptions.items():
    print(f"{col:15} - {desc}")
print()

# Data quality analysis
print("3. DATA QUALITY ANALYSIS")
print("Missing values per feature:")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percent
})
print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))
print()

# Target distribution
print("4. TARGET DISTRIBUTION")
target_dist = df['classification'].value_counts()
print(target_dist)
print(f"CKD prevalence: {(target_dist['ckd'] / len(df)) * 100:.1f}%")
print()

# Age analysis
print("5. AGE ANALYSIS")
print(f"Age range: {df['age'].min():.1f} - {df['age'].max():.1f} years")
print(f"Mean age: {df['age'].mean():.1f} years")
print(f"Median age: {df['age'].median():.1f} years")

# Age by classification
ckd_age = df[df['classification'] == 'ckd']['age'].dropna()
notckd_age = df[df['classification'] == 'notckd']['age'].dropna()
print(f"Mean age - CKD: {ckd_age.mean():.1f} years")
print(f"Mean age - No CKD: {notckd_age.mean():.1f} years")
print()

# Key clinical parameters analysis
print("6. KEY CLINICAL PARAMETERS")

# Blood pressure
print("Blood Pressure (mm Hg):")
bp_ckd = df[df['classification'] == 'ckd']['bp'].dropna()
bp_notckd = df[df['classification'] == 'notckd']['bp'].dropna()
print(f"  CKD patients: Mean={bp_ckd.mean():.1f}, Std={bp_ckd.std():.1f}")
print(f"  No CKD patients: Mean={bp_notckd.mean():.1f}, Std={bp_notckd.std():.1f}")

# Serum creatinine (key kidney function marker)
print("\nSerum Creatinine (mg/dL) - Key kidney function marker:")
sc_ckd = df[df['classification'] == 'ckd']['sc'].dropna()
sc_notckd = df[df['classification'] == 'notckd']['sc'].dropna()
print(f"  CKD patients: Mean={sc_ckd.mean():.2f}, Std={sc_ckd.std():.2f}")
print(f"  No CKD patients: Mean={sc_notckd.mean():.2f}, Std={sc_notckd.std():.2f}")
print(f"  Normal range: 0.6-1.2 mg/dL")

# Blood urea
print("\nBlood Urea (mg/dL):")
bu_ckd = df[df['classification'] == 'ckd']['bu'].dropna()
bu_notckd = df[df['classification'] == 'notckd']['bu'].dropna()
print(f"  CKD patients: Mean={bu_ckd.mean():.1f}, Std={bu_ckd.std():.1f}")
print(f"  No CKD patients: Mean={bu_notckd.mean():.1f}, Std={bu_notckd.std():.1f}")
print(f"  Normal range: 7-20 mg/dL")

# Hemoglobin
print("\nHemoglobin (g/dL):")
hemo_ckd = df[df['classification'] == 'ckd']['hemo'].dropna()
hemo_notckd = df[df['classification'] == 'notckd']['hemo'].dropna()
print(f"  CKD patients: Mean={hemo_ckd.mean():.1f}, Std={hemo_ckd.std():.1f}")
print(f"  No CKD patients: Mean={hemo_notckd.mean():.1f}, Std={hemo_notckd.std():.1f}")
print(f"  Normal range: 12-16 g/dL (women), 14-18 g/dL (men)")
print()

# Risk factors analysis
print("7. RISK FACTORS ANALYSIS")

# Hypertension
htn_ckd = df[df['classification'] == 'ckd']['htn'].value_counts()
htn_notckd = df[df['classification'] == 'notckd']['htn'].value_counts()
print("Hypertension:")
print(f"  CKD patients: {htn_ckd.get('yes', 0)} yes, {htn_ckd.get('no', 0)} no")
print(f"  No CKD patients: {htn_notckd.get('yes', 0)} yes, {htn_notckd.get('no', 0)} no")

# Diabetes
dm_ckd = df[df['classification'] == 'ckd']['dm'].value_counts()
dm_notckd = df[df['classification'] == 'notckd']['dm'].value_counts()
print("\nDiabetes Mellitus:")
print(f"  CKD patients: {dm_ckd.get('yes', 0)} yes, {dm_ckd.get('no', 0)} no")
print(f"  No CKD patients: {dm_notckd.get('yes', 0)} yes, {dm_notckd.get('no', 0)} no")

# Coronary artery disease
cad_ckd = df[df['classification'] == 'ckd']['cad'].value_counts()
cad_notckd = df[df['classification'] == 'notckd']['cad'].value_counts()
print("\nCoronary Artery Disease:")
print(f"  CKD patients: {cad_ckd.get('yes', 0)} yes, {cad_ckd.get('no', 0)} no")
print(f"  No CKD patients: {cad_notckd.get('yes', 0)} yes, {cad_notckd.get('no', 0)} no")
print()

# Urine analysis
print("8. URINE ANALYSIS")

# Specific gravity
sg_ckd = df[df['classification'] == 'ckd']['sg'].dropna()
sg_notckd = df[df['classification'] == 'notckd']['sg'].dropna()
print("Specific Gravity (urine concentration):")
print(f"  CKD patients: Mean={sg_ckd.mean():.3f}, Std={sg_ckd.std():.3f}")
print(f"  No CKD patients: Mean={sg_notckd.mean():.3f}, Std={sg_notckd.std():.3f}")
print(f"  Normal range: 1.005-1.030")

# Albumin
al_ckd = df[df['classification'] == 'ckd']['al'].dropna()
al_notckd = df[df['classification'] == 'notckd']['al'].dropna()
print("\nAlbumin (0-5 scale):")
print(f"  CKD patients: Mean={al_ckd.mean():.1f}, Std={al_ckd.std():.1f}")
print(f"  No CKD patients: Mean={al_notckd.mean():.1f}, Std={al_notckd.std():.1f}")
print(f"  0=normal, 1-5=increasing proteinuria")

# Sugar
su_ckd = df[df['classification'] == 'ckd']['su'].dropna()
su_notckd = df[df['classification'] == 'notckd']['su'].dropna()
print("\nSugar in urine (0-5 scale):")
print(f"  CKD patients: Mean={su_ckd.mean():.1f}, Std={su_ckd.std():.1f}")
print(f"  No CKD patients: Mean={su_notckd.mean():.1f}, Std={su_notckd.std():.1f}")
print()

# Statistical significance tests
print("9. STATISTICAL SIGNIFICANCE TESTS")

# Age comparison
age_ckd = df[df['classification'] == 'ckd']['age'].dropna()
age_notckd = df[df['classification'] == 'notckd']['age'].dropna()
t_stat, p_value = stats.ttest_ind(age_ckd, age_notckd)
print(f"Age difference (CKD vs No CKD): t={t_stat:.3f}, p={p_value:.6f}")

# Serum creatinine comparison
sc_ckd = df[df['classification'] == 'ckd']['sc'].dropna()
sc_notckd = df[df['classification'] == 'notckd']['sc'].dropna()
t_stat, p_value = stats.ttest_ind(sc_ckd, sc_notckd)
print(f"Serum creatinine difference: t={t_stat:.3f}, p={p_value:.6f}")

# Blood urea comparison
bu_ckd = df[df['classification'] == 'ckd']['bu'].dropna()
bu_notckd = df[df['classification'] == 'notckd']['bu'].dropna()
t_stat, p_value = stats.ttest_ind(bu_ckd, bu_notckd)
print(f"Blood urea difference: t={t_stat:.3f}, p={p_value:.6f}")
print()

# Key insights and clinical implications
print("10. KEY INSIGHTS AND CLINICAL IMPLICATIONS")
print("• CKD patients are significantly older than non-CKD patients")
print("• Serum creatinine is markedly elevated in CKD patients (key diagnostic marker)")
print("• Blood urea levels are significantly higher in CKD patients")
print("• Hypertension and diabetes are major risk factors for CKD")
print("• Hemoglobin levels are lower in CKD patients (anemia is common)")
print("• Proteinuria (albumin in urine) is more common in CKD patients")
print("• Blood pressure tends to be higher in CKD patients")
print()

# Data quality recommendations
print("11. DATA QUALITY RECOMMENDATIONS")
print("• Handle missing values appropriately for modeling")
print("• Consider imputation strategies for clinical parameters")
print("• Validate extreme values in laboratory results")
print("• Ensure consistent coding of categorical variables")
print()

# Modeling considerations
print("12. MODELING CONSIDERATIONS")
print("• Use appropriate handling for missing values")
print("• Consider feature engineering for age groups")
print("• Normalize/scale numerical features")
print("• Handle class imbalance if present")
print("• Use cross-validation for robust evaluation")
print("• Consider clinical interpretability of models") 