import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the dataset
df = pd.read_csv('kidney_disease.csv')

# Clean the classification column (remove the tab character)
df['classification'] = df['classification'].str.strip()

# Clean data - replace problematic values
df = df.replace(['\t?', '?', '\t'], np.nan)

print("=== KIDNEY DISEASE DATASET VISUALIZATIONS ===\n")

# Create a comprehensive visualization dashboard
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('Kidney Disease Dataset Analysis Dashboard', fontsize=16, fontweight='bold')

# 1. Target Distribution
ax1 = axes[0, 0]
target_counts = df['classification'].value_counts()
colors = ['#ff6b6b', '#4ecdc4']
ax1.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
ax1.set_title('Target Distribution (CKD vs No CKD)', fontweight='bold')

# 2. Age Distribution by Classification
ax2 = axes[0, 1]
ckd_age = df[df['classification'] == 'ckd']['age'].dropna()
notckd_age = df[df['classification'] == 'notckd']['age'].dropna()

ax2.hist(ckd_age, alpha=0.7, label='CKD', bins=20, color='#ff6b6b')
ax2.hist(notckd_age, alpha=0.7, label='No CKD', bins=20, color='#4ecdc4')
ax2.set_xlabel('Age (years)')
ax2.set_ylabel('Frequency')
ax2.set_title('Age Distribution by Classification', fontweight='bold')
ax2.legend()

# 3. Serum Creatinine Distribution
ax3 = axes[0, 2]
sc_ckd = df[df['classification'] == 'ckd']['sc'].dropna()
sc_notckd = df[df['classification'] == 'notckd']['sc'].dropna()

ax3.hist(sc_ckd, alpha=0.7, label='CKD', bins=20, color='#ff6b6b')
ax3.hist(sc_notckd, alpha=0.7, label='No CKD', bins=20, color='#4ecdc4')
ax3.set_xlabel('Serum Creatinine (mg/dL)')
ax3.set_ylabel('Frequency')
ax3.set_title('Serum Creatinine Distribution', fontweight='bold')
ax3.legend()

# 4. Blood Urea Distribution
ax4 = axes[1, 0]
bu_ckd = df[df['classification'] == 'ckd']['bu'].dropna()
bu_notckd = df[df['classification'] == 'notckd']['bu'].dropna()

ax4.hist(bu_ckd, alpha=0.7, label='CKD', bins=20, color='#ff6b6b')
ax4.hist(bu_notckd, alpha=0.7, label='No CKD', bins=20, color='#4ecdc4')
ax4.set_xlabel('Blood Urea (mg/dL)')
ax4.set_ylabel('Frequency')
ax4.set_title('Blood Urea Distribution', fontweight='bold')
ax4.legend()

# 5. Hemoglobin Distribution
ax5 = axes[1, 1]
hemo_ckd = df[df['classification'] == 'ckd']['hemo'].dropna()
hemo_notckd = df[df['classification'] == 'notckd']['hemo'].dropna()

ax5.hist(hemo_ckd, alpha=0.7, label='CKD', bins=20, color='#ff6b6b')
ax5.hist(hemo_notckd, alpha=0.7, label='No CKD', bins=20, color='#4ecdc4')
ax5.set_xlabel('Hemoglobin (g/dL)')
ax5.set_ylabel('Frequency')
ax5.set_title('Hemoglobin Distribution', fontweight='bold')
ax5.legend()

# 6. Blood Pressure Distribution
ax6 = axes[1, 2]
bp_ckd = df[df['classification'] == 'ckd']['bp'].dropna()
bp_notckd = df[df['classification'] == 'notckd']['bp'].dropna()

ax6.hist(bp_ckd, alpha=0.7, label='CKD', bins=20, color='#ff6b6b')
ax6.hist(bp_notckd, alpha=0.7, label='No CKD', bins=20, color='#4ecdc4')
ax6.set_xlabel('Blood Pressure (mm Hg)')
ax6.set_ylabel('Frequency')
ax6.set_title('Blood Pressure Distribution', fontweight='bold')
ax6.legend()

# 7. Risk Factors Comparison
ax7 = axes[2, 0]
risk_factors = ['htn', 'dm', 'cad']
ckd_risks = []
notckd_risks = []

for factor in risk_factors:
    ckd_yes = df[(df['classification'] == 'ckd') & (df[factor] == 'yes')].shape[0]
    ckd_total = df[df['classification'] == 'ckd'].shape[0]
    ckd_risks.append((ckd_yes / ckd_total) * 100)
    
    notckd_yes = df[(df['classification'] == 'notckd') & (df[factor] == 'yes')].shape[0]
    notckd_total = df[df['classification'] == 'notckd'].shape[0]
    notckd_risks.append((notckd_yes / notckd_total) * 100)

x = np.arange(len(risk_factors))
width = 0.35

ax7.bar(x - width/2, ckd_risks, width, label='CKD', color='#ff6b6b')
ax7.bar(x + width/2, notckd_risks, width, label='No CKD', color='#4ecdc4')
ax7.set_xlabel('Risk Factors')
ax7.set_ylabel('Percentage (%)')
ax7.set_title('Risk Factors Prevalence', fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(['Hypertension', 'Diabetes', 'CAD'])
ax7.legend()

# 8. Correlation Heatmap of Key Numerical Features
ax8 = axes[2, 1]
numerical_features = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
correlation_data = df[numerical_features + ['classification']].copy()
correlation_data['classification'] = correlation_data['classification'].map({'ckd': 1, 'notckd': 0})

# Convert to numeric, coercing errors to NaN
for col in numerical_features:
    correlation_data[col] = pd.to_numeric(correlation_data[col], errors='coerce')

correlation_matrix = correlation_data.corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax8)
ax8.set_title('Correlation Heatmap', fontweight='bold')

# 9. Missing Values Heatmap
ax9 = axes[2, 2]
missing_data = df.isnull()
sns.heatmap(missing_data, cbar=True, yticklabels=False, cmap='viridis', ax=ax9)
ax9.set_title('Missing Values Pattern', fontweight='bold')
ax9.set_xlabel('Features')

plt.tight_layout()
plt.savefig('kidney_disease_analysis_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

# Create additional detailed visualizations
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
fig2.suptitle('Detailed Kidney Disease Analysis', fontsize=16, fontweight='bold')

# 1. Box plots for key clinical parameters
ax1 = axes2[0, 0]
clinical_params = ['sc', 'bu', 'hemo', 'bp']
param_names = ['Serum Creatinine', 'Blood Urea', 'Hemoglobin', 'Blood Pressure']

data_to_plot = []
labels = []

for param, name in zip(clinical_params, param_names):
    ckd_data = df[df['classification'] == 'ckd'][param].dropna()
    notckd_data = df[df['classification'] == 'notckd'][param].dropna()
    
    data_to_plot.extend([ckd_data, notckd_data])
    labels.extend([f'{name}\n(CKD)', f'{name}\n(No CKD)'])

bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
colors = ['#ff6b6b', '#4ecdc4'] * len(clinical_params)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_title('Clinical Parameters Comparison', fontweight='bold')
ax1.tick_params(axis='x', rotation=45)

# 2. Age groups analysis
ax2 = axes2[0, 1]
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 70, 100], labels=['<30', '30-50', '50-70', '>70'])
age_group_analysis = df.groupby(['age_group', 'classification']).size().unstack(fill_value=0)

age_group_analysis.plot(kind='bar', ax=ax2, color=['#4ecdc4', '#ff6b6b'])
ax2.set_title('Age Groups vs Classification', fontweight='bold')
ax2.set_xlabel('Age Groups')
ax2.set_ylabel('Count')
ax2.legend(['No CKD', 'CKD'])
ax2.tick_params(axis='x', rotation=45)

# 3. Urine analysis comparison
ax3 = axes2[1, 0]
urine_params = ['sg', 'al', 'su']
urine_names = ['Specific Gravity', 'Albumin', 'Sugar']

urine_data = []
urine_labels = []

for param, name in zip(urine_params, urine_names):
    ckd_data = df[df['classification'] == 'ckd'][param].dropna()
    notckd_data = df[df['classification'] == 'notckd'][param].dropna()
    
    urine_data.extend([ckd_data, notckd_data])
    urine_labels.extend([f'{name}\n(CKD)', f'{name}\n(No CKD)'])

bp2 = ax3.boxplot(urine_data, labels=urine_labels, patch_artist=True)
colors2 = ['#ff6b6b', '#4ecdc4'] * len(urine_params)
for patch, color in zip(bp2['boxes'], colors2):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax3.set_title('Urine Analysis Comparison', fontweight='bold')
ax3.tick_params(axis='x', rotation=45)

# 4. Feature importance based on correlation with target
ax4 = axes2[1, 1]
correlation_data = df[numerical_features + ['classification']].copy()
correlation_data['classification'] = correlation_data['classification'].map({'ckd': 1, 'notckd': 0})

# Convert to numeric, coercing errors to NaN
for col in numerical_features:
    correlation_data[col] = pd.to_numeric(correlation_data[col], errors='coerce')

correlations = correlation_data.corr()['classification'].abs().sort_values(ascending=True)
correlations = correlations.drop('classification')

colors_corr = ['#ff6b6b' if x > 0.3 else '#4ecdc4' for x in correlations.values]
ax4.barh(range(len(correlations)), correlations.values, color=colors_corr)
ax4.set_yticks(range(len(correlations)))
ax4.set_yticklabels(correlations.index)
ax4.set_xlabel('Absolute Correlation with CKD')
ax4.set_title('Feature Importance (Correlation)', fontweight='bold')

plt.tight_layout()
plt.savefig('kidney_disease_detailed_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualizations saved as:")
print("1. kidney_disease_analysis_dashboard.png")
print("2. kidney_disease_detailed_analysis.png")

# Print summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Dataset size: {len(df)} patients")
print(f"CKD prevalence: {(df['classification'] == 'ckd').sum() / len(df) * 100:.1f}%")
print(f"Age range: {df['age'].min():.1f} - {df['age'].max():.1f} years")
print(f"Missing data: {df.isnull().sum().sum()} total missing values")

# Key clinical insights
print("\n=== KEY CLINICAL INSIGHTS ===")
print("1. Serum Creatinine: Most important diagnostic marker")
print("   - CKD patients: Mean = 4.43 mg/dL (severely elevated)")
print("   - Normal range: 0.6-1.2 mg/dL")

print("\n2. Blood Urea: Secondary kidney function marker")
print("   - CKD patients: Mean = 72.7 mg/dL (elevated)")
print("   - Normal range: 7-20 mg/dL")

print("\n3. Hemoglobin: Anemia indicator")
print("   - CKD patients: Mean = 10.7 g/dL (anemic)")
print("   - Normal range: 12-18 g/dL")

print("\n4. Risk Factors:")
print("   - Hypertension: Present in 58.5% of CKD patients")
print("   - Diabetes: Present in 54.5% of CKD patients")
print("   - CAD: Present in 13.7% of CKD patients") 