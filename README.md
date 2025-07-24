# AI-Based Early Detection of Chronic Kidney Disease (CKD) in Asthma Patients

This project is part of my Master's Major Research Project (MRP) at Toronto Metropolitan University. It explores the use of machine learning to detect early signs of Chronic Kidney Disease (CKD) in patients with asthma, using clinical data from the publicly available MIMIC-IV dataset.

## 🔍 Purpose

Asthma patients are at a heightened risk of developing chronic kidney disease, yet early detection is often missed. This project aims to:

- **Identify asthma patients** in the MIMIC-IV database.
- **Analyze clinical biomarkers and comorbidities** linked to kidney function.
- **Build predictive models** that can flag early-stage CKD using machine learning.
- **Support proactive diagnosis and treatment** to improve long-term health outcomes.

## 🧠 Methodology

1. **Data Source:** MIMIC-IV clinical database (v2.2)
2. **Target Group:** Adult asthma patients with and without CKD
3. **Features:** Age, gender, lab test results (e.g., creatinine, BUN, GFR), comorbidities (e.g., hypertension, diabetes)
4. **ML Models Used:** Logistic Regression, Random Forest, XGBoost, SVM, etc.
5. **Validation Techniques:** Train-test split, stratified k-fold cross-validation, and SMOTE for class imbalance
6. **Performance Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC

## 📊 Key Findings

- Biomarkers like creatinine, BUN, and GFR were strongly associated with early signs of CKD.
- Tree-based models (Random Forest, XGBoost) outperformed others in prediction accuracy.
- The models can aid in **early risk screening**, especially in settings with limited access to nephrology specialists.

## 📁 Dataset

- **MIMIC-IV**: A freely available critical care database maintained by the MIT Lab for Computational Physiology.
- Access it at: [https://physionet.org/content/mimiciv/](https://physionet.org/content/mimiciv/)
- Dataset use complies with PhysioNet credentialing and ethical guidelines.

## 👩‍💻 Author

**Momna Ali**  
Master of Science in Data Science and Analytics  
Toronto Metropolitan University  
Summer 2025

## 📘 Acknowledgments

- MIT Lab for Computational Physiology (for MIMIC-IV)
- TMU Faculty of Science & MRP Advisors
- Open-source contributors to scikit-learn, XGBoost, pandas, matplotlib, and related libraries

## ⚠️ Disclaimer

This project is for academic research only and does not constitute medical advice or diagnosis.
