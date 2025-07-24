#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load Dataset
patients = pd.read_csv('hosp/patients.csv')
admissions = pd.read_csv('hosp/admissions.csv')
diagnoses_icd = pd.read_csv('hosp/diagnoses_icd.csv')
d_icd_diagnoses = pd.read_csv('hosp/d_icd_diagnoses.csv')

# Print column names and data types for each file
print("=== patients.csv ===")
print(patients.dtypes)
print("\n")

print("=== admissions.csv ===")
print(admissions.dtypes)
print("\n")

print("=== diagnoses_icd.csv ===")
print(diagnoses_icd.dtypes)
print("\n")

print("=== d_icd_diagnoses.csv ===")
print(d_icd_diagnoses.dtypes)


# Step 1: Load and Filter for Asthma Patients in your EDA.

# In[2]:


import pandas as pd

# Step 1: Load datasets
patients = pd.read_csv('hosp/patients.csv')
admissions = pd.read_csv('hosp/admissions.csv')
diagnoses_icd = pd.read_csv('hosp/diagnoses_icd.csv')
d_icd_diagnoses = pd.read_csv('hosp/d_icd_diagnoses.csv')

# Step 2: Merge ICD descriptions into diagnoses
diagnoses_full = diagnoses_icd.merge(d_icd_diagnoses, on=['icd_code', 'icd_version'], how='left')

# Step 3: Filter for asthma-related diagnoses (ICD-9: 493.*, ICD-10: J45*)
asthma_patients = diagnoses_full[
    diagnoses_full['icd_code'].str.startswith(('493', 'J45'))
]

# Check unique matches
print("Unique ICD asthma matches:", asthma_patients['long_title'].unique())

# Step 4: Get unique subject_ids with asthma
asthma_subject_ids = asthma_patients['subject_id'].unique()

# Step 5: Subset demographics and admissions for these patients
asthma_demo = patients[patients['subject_id'].isin(asthma_subject_ids)]
asthma_adm = admissions[admissions['subject_id'].isin(asthma_subject_ids)]

# Step 6: Merge for enriched patient table
asthma_cohort = asthma_demo.merge(asthma_adm, on='subject_id', how='left')

# Preview
print("Asthma cohort shape:", asthma_cohort.shape)
asthma_cohort.head()


# Step 2: Integrate CKD lab biomarkers

# In[3]:


import pandas as pd

# Load Dataset
labevents = pd.read_csv('hosp/labevents.csv', nrows=5)
d_labitems = pd.read_csv('hosp/d_labitems.csv')

# Print column names and data types for each file
print("=== labevents.csv ===")
print(labevents.dtypes)
print("\n")

print("=== d_labitems.csv ===")
print(d_labitems.dtypes)
print("\n")


# Get CKD Lab Test itemids

# In[4]:


import pandas as pd

# Load lab item definitions
d_labitems = pd.read_csv('hosp/d_labitems.csv')

# Identify CKD-relevant test names
ckd_keywords = ['creatinine', 'bun', 'urea', 'egfr', 'protein', 'albumin', 'hemoglobin']

# Filter lab items based on name match
ckd_labitems = d_labitems[d_labitems['label'].str.lower().str.contains('|'.join(ckd_keywords), na=False)]

# Extract list of relevant itemids
ckd_itemids = ckd_labitems['itemid'].unique()

# Display for verification
print("CKD Lab Tests Identified:")
print(ckd_labitems[['itemid', 'label']])


# Chunked Filtering of labevents.csv

# In[5]:


import pandas as pd

# Load filtered asthma subject_ids from Step 1
asthma_subject_ids = set(asthma_cohort['subject_id'])

# File path to large labevents file
labevents_path = 'hosp/labevents.csv'

# Set chunk size
chunksize = 10**6

# Empty list to collect relevant rows
filtered_labs = []

# Read in chunks
for chunk in pd.read_csv(labevents_path, chunksize=chunksize):
    # Keep only rows matching asthma patients AND CKD lab tests
    mask = (chunk['subject_id'].isin(asthma_subject_ids)) & (chunk['itemid'].isin(ckd_itemids))
    filtered = chunk.loc[mask].dropna(subset=['valuenum'])  # remove missing values

    # Append filtered chunk
    filtered_labs.append(filtered)

# Combine all filtered rows into a single DataFrame
asthma_lab_events_clean = pd.concat(filtered_labs, ignore_index=True)

# Merge labels from d_labitems for clarity
asthma_lab_events_clean = asthma_lab_events_clean.merge(
    d_labitems[['itemid', 'label']], on='itemid', how='left'
)

# Preview result
print("Total CKD-related lab events for asthma patients:", asthma_lab_events_clean.shape)
asthma_lab_events_clean[['subject_id', 'label', 'valuenum', 'charttime']].head()


# Step 3: Medication + Service Context for Asthma Cohort

# In[6]:


# Load Dataset
prescriptions = pd.read_csv('hosp/prescriptions.csv', low_memory=False)
services = pd.read_csv('hosp/services.csv')


# Print column names and data types for each file
print("=== prescriptions.csv ===")
print(prescriptions.dtypes)
print("\n")

print("=== services.csv ===")
print(services.dtypes)


# Chunked Extraction of Steroid & NSAID Prescriptions

# In[7]:


import pandas as pd

# Define steroid and NSAID keywords
steroid_keywords = ['prednisone', 'prednisolone', 'methylprednisolone', 'dexamethasone', 'hydrocortisone']
nsaid_keywords = ['ibuprofen', 'naproxen', 'diclofenac', 'celecoxib', 'ketorolac']

# Convert subject_ids to set for fast lookup
asthma_ids_set = set(asthma_cohort['subject_id'])

# Set path and chunk size
rx_path = 'hosp/prescriptions.csv'
chunksize = 10**5  # Tune based on your system

# Container to store filtered results
filtered_rx_chunks = []

# Read in chunks
for chunk in pd.read_csv(rx_path, chunksize=chunksize, low_memory=False):
    chunk['drug_lower'] = chunk['drug'].astype(str).str.lower()

    # Filter by asthma patients
    chunk = chunk[chunk['subject_id'].isin(asthma_ids_set)]

    # Filter by drug keywords
    is_steroid = chunk['drug_lower'].str.contains('|'.join(steroid_keywords))
    is_nsaid = chunk['drug_lower'].str.contains('|'.join(nsaid_keywords))

    filtered = chunk[is_steroid | is_nsaid].copy()
    filtered['is_steroid'] = is_steroid
    filtered['is_nsaid'] = is_nsaid

    # Append
    filtered_rx_chunks.append(filtered)

# Concatenate all filtered chunks
asthma_rx_filtered = pd.concat(filtered_rx_chunks, ignore_index=True)

# Preview results
print("Filtered prescription records:", asthma_rx_filtered.shape)
asthma_rx_filtered[['subject_id', 'drug', 'is_steroid', 'is_nsaid', 'starttime']].head()


# Extract Service Context for Admissions

# In[8]:


# Load services.csv
services = pd.read_csv('hosp/services.csv')

# Filter to asthma patients
asthma_services = services[services['subject_id'].isin(asthma_cohort['subject_id'])]

# Merge with admission context
asthma_service_context = asthma_cohort.merge(asthma_services, on=['subject_id', 'hadm_id'], how='left')

# Check frequency of current services
print("Top current services during admission:")
print(asthma_service_context['curr_service'].value_counts().head())


# Step 4: Feature Engineering for Asthma-CKD Prediction

# Step 4A: Aggregate CKD Biomarkers Per Patient

# In[9]:


# Group and summarize CKD labs per patient
biomarker_summary = asthma_lab_events_clean.groupby(['subject_id', 'label'])['valuenum'].agg(['mean', 'min', 'max']).reset_index()

# Pivot to wide format: one row per patient
biomarker_pivot = biomarker_summary.pivot(index='subject_id', columns='label', values='mean').reset_index()

# Rename columns for easier access
biomarker_pivot.columns.name = None
biomarker_pivot = biomarker_pivot.rename(columns=lambda x: x.lower().replace(' ', '_') if isinstance(x, str) else x)

# Preview
biomarker_pivot.head()


# Step 4B: Create Medication Exposure Flags

# In[10]:


# Get binary flags per patient
rx_flags = asthma_rx_filtered.groupby('subject_id')[['is_steroid', 'is_nsaid']].max().reset_index()

# Preview
rx_flags.head()


# Step 4C: Encode Admission Service Info

# In[11]:


# Get most common current service per subject
top_service = asthma_service_context.groupby('subject_id')['curr_service'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown').reset_index()

# Optional: one-hot encode services
top_service_dummies = pd.get_dummies(top_service['curr_service'], prefix='service')
top_service_encoded = pd.concat([top_service['subject_id'], top_service_dummies], axis=1)

# Preview
top_service_encoded.head()


# Step 4D: Merge All Features into One Master Table

# In[12]:


# Start with demographic base
features_df = asthma_cohort[['subject_id', 'gender', 'anchor_age']].drop_duplicates()

# Merge engineered features
features_df = features_df.merge(biomarker_pivot, on='subject_id', how='left')
features_df = features_df.merge(rx_flags, on='subject_id', how='left')
features_df = features_df.merge(top_service_encoded, on='subject_id', how='left')

# Fill missing binary flags
features_df[['is_steroid', 'is_nsaid']] = features_df[['is_steroid', 'is_nsaid']].fillna(0)

# Final feature table preview
print("Final Feature Table:", features_df.shape)
features_df.head()


# eGFR-based CKD stages

# In[13]:


# CKD-EPI 2009 eGFR Computation
import numpy as np

# Normalize gender values
features_df['gender'] = features_df['gender'].str.upper().str.strip()

# Filter valid rows
features_df = features_df[
    (features_df['gender'].isin(['M', 'F'])) &
    (features_df['creatinine'].notna()) &
    (features_df['anchor_age'].notna())
]

# Define CKD-EPI formula
def compute_egfr(row):
    Scr = row['creatinine']
    age = row['anchor_age']
    gender = row['gender']

    if gender == 'F':
        k = 0.7
        alpha = -0.329
        multiplier = 1.018
    else:
        k = 0.9
        alpha = -0.411
        multiplier = 1.0

    min_ratio = min(Scr / k, 1)
    max_ratio = max(Scr / k, 1)

    return 141 * (min_ratio ** alpha) * (max_ratio ** -1.209) * (0.993 ** age) * multiplier

# Apply to DataFrame
features_df['egfr_ckd_epi'] = features_df.apply(compute_egfr, axis=1)

# Preview results
features_df[['subject_id', 'creatinine', 'anchor_age', 'gender', 'egfr_ckd_epi']].head()


# Step 5: Visualizations

# Step 5A: Creatinine Distribution by Steroid Use

# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.kdeplot(data=features_df, x='creatinine', hue='is_steroid', fill=True, common_norm=False, alpha=0.5, palette='Set2')
plt.title('Creatinine Levels by Steroid Exposure')
plt.xlabel('Mean Creatinine')
plt.ylabel('Density')
plt.legend(title='Steroid Use', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()


# Step 5B: Boxplot: Urea Nitrogen by NSAID Exposure

# In[15]:


plt.figure(figsize=(6, 5))
sns.boxplot(x='is_nsaid', y='urea_nitrogen', data=features_df, hue='is_nsaid', palette='pastel', legend=False)
plt.title('Urea Nitrogen by NSAID Exposure')
plt.xlabel('NSAID Use (0=No, 1=Yes)')
plt.ylabel('Mean Urea Nitrogen')
plt.tight_layout()
plt.show()


# eGFR Visualizations

# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of eGFR (CKD-EPI)
plt.figure(figsize=(8, 5))
sns.histplot(data=features_df, x='egfr_ckd_epi', bins=40, kde=True, color='skyblue')
plt.axvline(60, color='red', linestyle='--', label='CKD Stage 3 Threshold')
plt.title('Distribution of eGFR (CKD-EPI) among Asthma Patients')
plt.xlabel('eGFR (mL/min/1.73m²)')
plt.ylabel('Patient Count')
plt.legend()
plt.tight_layout()
plt.show()

# eGFR by Steroid Use
plt.figure(figsize=(7, 5))
sns.boxplot(x='is_steroid', y='egfr_ckd_epi', data=features_df, palette='Set2')
plt.title('eGFR (CKD-EPI) by Steroid Exposure')
plt.xlabel('Steroid Use (0 = No, 1 = Yes)')
plt.ylabel('eGFR (mL/min/1.73m²)')
plt.tight_layout()
plt.show()

# eGFR vs. Age
plt.figure(figsize=(7, 5))
sns.scatterplot(data=features_df, x='anchor_age', y='egfr_ckd_epi', hue='is_nsaid', alpha=0.7, palette='coolwarm')
plt.axhline(60, color='gray', linestyle='--')
plt.title('eGFR vs. Age (Colored by NSAID Exposure)')
plt.xlabel('Age')
plt.ylabel('eGFR (CKD-EPI)')
plt.legend(title='NSAID Use')
plt.tight_layout()
plt.show()

# CKD Stage Distribution
def ckd_stage(egfr):
    if egfr >= 90:
        return 'Stage 1 (Normal)'
    elif egfr >= 60:
        return 'Stage 2 (Mild)'
    elif egfr >= 30:
        return 'Stage 3 (Moderate)'
    elif egfr >= 15:
        return 'Stage 4 (Severe)'
    else:
        return 'Stage 5 (Failure)'

features_df['ckd_stage'] = features_df['egfr_ckd_epi'].apply(ckd_stage)

ckd_counts = features_df['ckd_stage'].value_counts().sort_index()
plt.figure(figsize=(6, 6))
plt.pie(ckd_counts, labels=ckd_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set3"))
plt.title('CKD Stage Distribution Based on eGFR (CKD-EPI)')
plt.tight_layout()
plt.show()


# Step 5C: Correlation Heatmap for CKD Biomarkers

# Filter for Key CKD Biomarkers Only

# In[17]:


ckd_core_features = [
    'creatinine', 'creatinine_serum', 'urea_nitrogen',
    'albumin', 'albumin/creatinine,_urine',
    'protein', 'total_protein,_urine',
    'hemoglobin', 'egfr_ckd_epi'
]

# Subset + drop missing ones
core_cols = [col for col in ckd_core_features if col in features_df.columns]
corr_core = features_df[core_cols].corr()

# Plot cleaned heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_core, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Correlation Heatmap: Core CKD Biomarkers')
plt.tight_layout()
plt.show()


# Step 5D: Creatinine vs. Age (Colored by Steroid Use)

# In[18]:


plt.figure(figsize=(7, 5))
sns.scatterplot(data=features_df, x='anchor_age', y='creatinine', hue='is_steroid', alpha=0.7)
plt.title('Creatinine vs. Age by Steroid Use')
plt.xlabel('Age')
plt.ylabel('Creatinine')
plt.tight_layout()
plt.show()


# Step 5E: Clinical Service Line Distribution

# In[19]:


service_cols = [col for col in features_df.columns if col.startswith('service_')]
service_distribution = features_df[service_cols].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=service_distribution.values, y=service_distribution.index, palette='muted')
plt.title('Distribution of Clinical Services among Asthma Patients')
plt.xlabel('Patient Count')
plt.ylabel('Service Line')
plt.tight_layout()
plt.show()


# Step 6: Predictive Modeling — Early CKD Detection in Asthma Patients

# Step 6A: Define the Target Variable (Binary CKD Label)

# In[20]:


# Define binary target: 1 = CKD Stage 3 or worse (eGFR < 60), 0 = Normal/Mild
features_df['ckd_label'] = features_df['egfr_ckd_epi'].apply(lambda x: 1 if x < 60 else 0)

# Check distribution
features_df['ckd_label'].value_counts()


# Step 6B: Select Features + Train/Test Split

# In[21]:


from sklearn.model_selection import train_test_split

# Drop columns not needed
drop_cols = ['subject_id', 'egfr_ckd_epi', 'ckd_stage', 'ckd_label']
feature_cols = [col for col in features_df.columns if col not in drop_cols and features_df[col].dtype in ['float64', 'int64']]

# Define X and y
X = features_df[feature_cols].fillna(0)  # Fill NA for model input
y = features_df['ckd_label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# Step 6C: Train Baseline Models (Random Forest, XGBoost)

# Sanitize Column Names

# In[30]:


X_train.columns = X_train.columns.str.replace(r"[\[\]<>]", "", regex=True)
X_train.columns = X_train.columns.str.replace(r"[ ,]", "_", regex=True)

X_test.columns = X_test.columns.str.replace(r"[\[\]<>]", "", regex=True)
X_test.columns = X_test.columns.str.replace(r"[ ,]", "_", regex=True)


# In[31]:


X_train.columns[X_train.columns.duplicated()].tolist()


# In[32]:


X_train = X_train.loc[:, ~X_train.columns.duplicated()]
X_test = X_test.loc[:, ~X_test.columns.duplicated()]


# In[25]:


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)


# Step 6D: Evaluate Model Performance

# In[26]:


from sklearn.metrics import classification_report, roc_auc_score

# Random Forest
rf_preds = rf_model.predict(X_test)
rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1])
print("Random Forest:\n", classification_report(y_test, rf_preds))
print("AUC:", rf_auc)

# XGBoost
xgb_preds = xgb_model.predict(X_test)
xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1])
print("XGBoost:\n", classification_report(y_test, xgb_preds))
print("AUC:", xgb_auc)


# Step 6E: Random Forest Feature Importance

# In[27]:


import matplotlib.pyplot as plt
import pandas as pd

# Corrected: Use X_train.columns
feat_imp = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(15)

plt.figure(figsize=(8, 5))
feat_imp.plot(kind='barh')
plt.title("Top 15 Features - Random Forest")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# (Part 2): XGBoost Feature Importance Plot

# In[28]:


import pandas as pd
import matplotlib.pyplot as plt

# Extract feature importances from XGBoost
xgb_importances = pd.Series(xgb_model.feature_importances_, index=X_train.columns)

# Get top 15 most important features
top_xgb_features = xgb_importances.sort_values(ascending=False).head(15)

# Plot
plt.figure(figsize=(8, 5))
top_xgb_features.plot(kind='barh')
plt.title("Top 15 Features - XGBoost")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# In[29]:


print("Target variable:", y.name)
print("Feature variables:", X_train.columns.tolist())


# Experiment 2 - Stratified K-Fold Cross-Validation vs. Train-Test Split

# In[55]:


xgb_model_cv = XGBClassifier(
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    tree_method='auto'  # <- Prevents automatic QuantileDMatrix fallback
)


# In[56]:


X_cleaned_fixed.columns = [str(col).replace('[','').replace(']','').replace('<','_lt_').replace('>','_gt_') for col in X_cleaned_fixed.columns]


# In[60]:


from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_auc_scores = cross_val_score(xgb_model_cv, X_cleaned_fixed.to_numpy(), y.to_numpy(), cv=cv, scoring='roc_auc', n_jobs=-1)
#The conversion to NumPy arrays was required to resolve a known compatibility issue between XGBoost and cross-validation in certain environments.

print("XGBoost AUC Scores (CV):", xgb_auc_scores)
print("Mean AUC:", xgb_auc_scores.mean())


# In[65]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Define Stratified K-Fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Reinitialize Random Forest
rf_model_cv = RandomForestClassifier(random_state=42)

# Run cross-validation
rf_auc_scores = cross_val_score(rf_model_cv, X_cleaned_fixed.to_numpy(), y.to_numpy(),
                                cv=cv, scoring='roc_auc', n_jobs=-1)

# Print results
print("Random Forest AUC Scores (CV):", rf_auc_scores)
print("Mean AUC:", round(rf_auc_scores.mean(), 4))


# Visualize

# In[69]:


import matplotlib.pyplot as plt
import numpy as np

rf_auc_scores = [0.99038571, 0.98976648, 0.99077589, 0.98892923, 0.99009388]  # Random Forest scores
xgb_auc_scores = [0.99207265, 0.99299741, 0.99302301, 0.99233833, 0.99239842]  # XGBoost scores

# Calculate mean AUCs
rf_mean = np.mean(rf_auc_scores)
xgb_mean = np.mean(xgb_auc_scores)

# Create bar chart
models = ['Random Forest', 'XGBoost']
mean_aucs = [rf_mean, xgb_mean]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, mean_aucs)

# Annotate bars with exact values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.002, round(yval, 3), 
             ha='center', va='bottom', fontsize=10)

plt.title('Experiment 2: Mean AUC (Stratified K-Fold CV)')
plt.ylabel('Mean AUC')
plt.ylim(0.85, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Experiment 3: Imbalance Handling with SMOTE

# In[63]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Step 1: Split the data
X_train, X_test, y_train, y_test = train_test_split(X_cleaned_fixed, y, test_size=0.2, random_state=42, stratify=y)

# Step 2: Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Convert to numpy arrays for compatibility with XGBoost
X_train_np = X_train_smote.to_numpy()
y_train_np = y_train_smote.to_numpy()
X_test_np = X_test.to_numpy()

# Train models
rf_smote_model = RandomForestClassifier(random_state=42)
xgb_smote_model = XGBClassifier(eval_metric='logloss', random_state=42)

rf_smote_model.fit(X_train_smote, y_train_smote)  # RandomForest can handle DataFrame
xgb_smote_model.fit(X_train_np, y_train_np)       # XGBoost requires NumPy

# Evaluate
rf_pred = rf_smote_model.predict(X_test)
xgb_pred = xgb_smote_model.predict(X_test_np)

rf_auc = roc_auc_score(y_test, rf_smote_model.predict_proba(X_test)[:, 1])
xgb_auc = roc_auc_score(y_test, xgb_smote_model.predict_proba(X_test_np)[:, 1])

print("Random Forest after SMOTE:\n")
print(classification_report(y_test, rf_pred))
print(f"AUC: {rf_auc}\n")

print("XGBoost after SMOTE:\n")
print(classification_report(y_test, xgb_pred))
print(f"AUC: {xgb_auc}")


# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# Models and classes
models = ['Random Forest', 'XGBoost']
classes = ['Class 0', 'Class 1']
metrics = ['Precision', 'Recall', 'F1-score']

# Metric values from results
scores = {
    'Random Forest': {
        'Precision': [0.98, 0.85],
        'Recall':    [0.96, 0.92],
        'F1-score':  [0.97, 0.89]
    },
    'XGBoost': {
        'Precision': [0.98, 0.87],
        'Recall':    [0.96, 0.93],
        'F1-score':  [0.97, 0.90]
    }
}

# Plot setup
n_classes = len(classes)
x = np.arange(n_classes)           # the label locations
width = 0.35                       # the width of the bars

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    # extract values for each model
    rf_vals  = scores['Random Forest'][metric]
    xgb_vals = scores['XGBoost'][metric]

    # plot bars
    ax.bar(x - width/2, rf_vals,  width, label='Random Forest')
    ax.bar(x + width/2, xgb_vals, width, label='XGBoost')

    # styling
    ax.set_title(f'{metric} by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_xlabel('Class')
    if idx == 0:
        ax.set_ylabel(metric)
    ax.legend()

plt.tight_layout()
plt.show()


# Appendix A — Hyperparameter Tuning
# 
# 🔍 Objective:
# To further optimize the predictive performance of both Random Forest and XGBoost classifiers by performing grid search-based hyperparameter tuning using stratified 5-fold cross-validation.

# In[29]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_cv = GridSearchCV(RandomForestClassifier(random_state=42),
                     param_grid=rf_grid,
                     scoring='roc_auc',
                     cv=5,
                     n_jobs=-1,
                     verbose=1)

rf_cv.fit(X_train, y_train)

print("Best Random Forest Params:", rf_cv.best_params_)
print("Best AUC:", rf_cv.best_score_)


# In[30]:


from xgboost import XGBClassifier

xgb_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_cv = GridSearchCV(XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
                      param_grid=xgb_grid,
                      scoring='roc_auc',
                      cv=5,
                      n_jobs=-1,
                      verbose=1)

xgb_cv.fit(X_train, y_train)

print("Best XGBoost Params:", xgb_cv.best_params_)
print("Best AUC:", xgb_cv.best_score_)


# Step 2: Visualization of Tuning Results

# In[31]:


import pandas as pd

# Convert GridSearch results to DataFrame
rf_results = pd.DataFrame(rf_cv.cv_results_)

# Only select useful columns for visualization
rf_results_filtered = rf_results[[
    'param_n_estimators', 
    'param_max_depth', 
    'param_min_samples_split', 
    'param_min_samples_leaf', 
    'mean_test_score'
]]


# In[32]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=rf_results_filtered,
    x='param_n_estimators',
    y='mean_test_score',
    hue='param_max_depth',
    marker='o'
)
plt.title('Random Forest AUC vs. n_estimators for Each max_depth')
plt.ylabel('Mean AUC (5-fold CV)')
plt.xlabel('Number of Estimators')
plt.legend(title='max_depth')
plt.tight_layout()
plt.show()


# In[33]:


# Pivot data to create heatmap grid
heatmap_data = rf_results_filtered.pivot_table(
    values='mean_test_score',
    index='param_max_depth',
    columns='param_n_estimators'
)

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".4f",
    cmap='YlGnBu',
    cbar_kws={'label': 'Mean AUC (5-fold CV)'}
)
plt.title('Random Forest AUC Heatmap: max_depth vs n_estimators')
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.tight_layout()
plt.show()


# XGBoost tuning visualizations

# In[34]:


# Create a DataFrame from XGBoost tuning results
xgb_results = pd.DataFrame(xgb_cv.cv_results_)

# Filter columns of interest
xgb_results_filtered = xgb_results[[
    'param_n_estimators',
    'param_max_depth',
    'param_learning_rate',
    'param_subsample',
    'param_colsample_bytree',
    'mean_test_score'
]]


# In[35]:


plt.figure(figsize=(10, 6))
sns.lineplot(
    data=xgb_results_filtered,
    x='param_n_estimators',
    y='mean_test_score',
    hue='param_max_depth',
    marker='o'
)
plt.title('XGBoost AUC vs. n_estimators for Each max_depth')
plt.ylabel('Mean AUC (5-fold CV)')
plt.xlabel('Number of Estimators')
plt.legend(title='max_depth')
plt.tight_layout()
plt.show()


# In[36]:


# Pivot for heatmap: mean AUC across learning_rate × max_depth
heatmap_data_xgb = xgb_results_filtered.pivot_table(
    values='mean_test_score',
    index='param_learning_rate',
    columns='param_max_depth'
)

plt.figure(figsize=(8, 6))
sns.heatmap(
    heatmap_data_xgb,
    annot=True,
    fmt=".4f",
    cmap='YlOrRd',
    cbar_kws={'label': 'Mean AUC (5-fold CV)'}
)
plt.title('XGBoost AUC Heatmap: learning_rate vs max_depth')
plt.xlabel('max_depth')
plt.ylabel('learning_rate')
plt.tight_layout()
plt.show()


# In[ ]:




