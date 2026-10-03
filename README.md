# 💳 Credit Card Fraud Detection — Production-Oriented ML Pipeline

> **From “Train a Model” → “Engineer an ML System”**

An end-to-end **Credit Card Fraud Detection system** designed beyond the traditional Kaggle workflow.

Instead of stopping at model training and accuracy, this project focuses on the complete ML lifecycle:

**Data Auditing → Modeling → Imbalance Handling → Feature Engineering → Threshold Optimization → Hyperparameter Tuning → Anomaly Detection → Explainability → Inference → Drift Monitoring**

---

## 🚀 Why This Project?

While exploring fraud detection implementations and Kaggle notebooks, I noticed a common pattern:

```text
EDA
 ↓
Preprocessing
 ↓
Train Model
 ↓
Accuracy / F1
 ↓
Done
```

But real-world fraud detection raises much harder questions:

* What should the **decision threshold** be?
* What is the business cost of a false negative?
* Is accuracy meaningful with extreme class imbalance?
* Can hyperparameters be systematically optimized?
* Does anomaly detection actually add useful signal?
* Why did the model classify a transaction as fraud?
* What happens when the data distribution changes?
* Can the same preprocessing pipeline be reused during inference?

This project was built to answer those questions.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │   Transaction Data  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Data Audit       │
                    │ Duplicates / Labels │
                    │ Class Distribution  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Preprocessing &     │
                    │ Train/Val/Test Split│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │ Time / Amount / V*  │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
       ┌─────────────────┐           ┌─────────────────┐
       │ Supervised ML   │           │ Anomaly Detection│
       │                 │           │                 │
       │ Logistic Reg.   │           │ Isolation Forest│
       │ Random Forest   │           └────────┬────────┘
       │ XGBoost         │                    │
       └────────┬────────┘                    │
                │                             │
                ▼                             ▼
       ┌──────────────────────────────────────────┐
       │          Model Evaluation                │
       │ PR-AUC / ROC-AUC / Precision / Recall   │
       └────────────────────┬─────────────────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Threshold Optimization│
                 │ Business Cost        │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Explainability       │
                 │ SHAP + Error Analysis│
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Production Inference │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Drift Monitoring     │
                 │ KS Statistics        │
                 └──────────────────────┘
```

---

# 📊 Dataset

**Credit Card Fraud Detection Dataset**

Original dataset:

* **284,807 transactions**
* **492 fraud transactions**
* Fraud rate: approximately **0.172%**
* 28 PCA-transformed features: `V1–V28`
* `Time`
* `Amount`
* `Class` → target

### Data Audit

The initial audit identified:

* **1,081 exact duplicate rows**
* Duplicate groups were checked for conflicting labels
* Duplicates were removed before modeling
* A corrected `Hour` feature was derived from `Time`

Final cleaned dataset:

```text
283,726 rows
32 features
```

Class distribution after cleaning:

```text
Genuine : 283,253
Fraud   :     473

Fraud Rate ≈ 0.167%
```

---

# 🔬 1. Data Splitting

A stratified split was used to preserve the extremely rare fraud class.

```text
                    283,726
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
       Train          Val          Test
      226,980        28,373       28,373
       378 fraud      48 fraud      47 fraud
```

The final test set remained completely isolated until final evaluation.

---

# 🤖 2. Model Progression

Rather than immediately selecting a complex model, the project established progressively stronger baselines.

### Logistic Regression

Used as the initial interpretable baseline.

```text
PR-AUC   : 0.6371
ROC-AUC  : 0.9315
F1       : 0.6667
Recall   : 54.17%
```

### Random Forest

```text
PR-AUC   : 0.7658
ROC-AUC  : 0.9226
F1       : 0.8193
Recall   : 70.83%
```

### XGBoost

```text
PR-AUC   : 0.8085
ROC-AUC  : 0.9657
F1       : 0.8434
Recall   : 72.92%
```

The progression demonstrated why **PR-AUC** was more informative than accuracy for this highly imbalanced problem.

---

# ⚖️ 3. Imbalanced Learning

Only around **0.17%** of transactions were fraudulent.

Therefore:

> **Accuracy alone can be dangerously misleading.**

Multiple approaches were evaluated:

```text
                 Imbalance Handling
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
  Class Weight      Oversampling        SMOTE
        │                │                │
        └────────────────┼────────────────┘
                         ▼
                  Compare PR-AUC
```

Approaches tested:

* Class weighting
* Random oversampling
* SMOTE
* Baseline XGBoost

The baseline XGBoost configuration was retained based on validation performance rather than forcing a more complicated sampling strategy.

---

# 🎯 4. Threshold Optimization

One of the most important engineering decisions in the project:

> **0.5 is not automatically the correct fraud threshold.**

The model produces a probability:

```text
P(Fraud | Transaction)
```

The system then converts that probability into a business decision.

```text
                Fraud Probability
                       │
                       ▼
                ┌─────────────┐
                │  Threshold  │
                │ Optimization│
                └──────┬──────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
           Genuine             Fraud
```

Thresholds were evaluated against:

* Precision
* Recall
* F1
* False positives
* False negatives
* Illustrative business costs

For example:

```text
Threshold     Precision     Recall
-----------------------------------
0.01           67.24%       81.25%
0.10           90.00%       75.00%
0.18           97.30%       75.00%
0.50          100.00%       72.92%
```

This demonstrates the fundamental fraud-detection trade-off:

**Lower threshold → catch more fraud, potentially more false alarms**

**Higher threshold → fewer false alarms, potentially more missed fraud**

---

# 🧬 5. Advanced Feature Engineering

The original dataset was expanded from **31 → 44 features**.

Additional features included:

### Transaction transformations

```text
LogAmount
TimeHours
HourSin
HourCos
```

### PCA-component statistics

```text
V_Mean
V_Std
V_Min
V_Max
V_AbsMean
V_AbsMax
V_L2
```

### Interaction features

```text
AmountPerHour
Amount_V_AbsMean
```

Feature engineering improved validation PR-AUC from approximately:

```text
0.8085 → 0.8146
```

---

# 🔎 6. Hyperparameter Optimization

Instead of relying on manually selected XGBoost parameters, **Optuna** was used with **TPE-based optimization**.

Search dimensions included:

```text
n_estimators
max_depth
learning_rate
subsample
colsample_bytree
min_child_weight
gamma
reg_alpha
reg_lambda
```

### Optimization Objective

```text
                    Optuna
                       │
                       ▼
                Generate Trial
                       │
                       ▼
                  Train XGBoost
                       │
                       ▼
                 Validation
                       │
                       ▼
                    PR-AUC
                       │
                       ▼
              Update Best Trial
                       │
                       ▼
                Next Trial
```

**30 trials** were completed.

Best validation PR-AUC:

```text
0.814246
```

The tuning result was very close to the feature-engineered baseline, which itself is an important finding:

> More hyperparameter search does not automatically produce a meaningful improvement.

---

# 🕵️ 7. Anomaly Detection

An **Isolation Forest** was evaluated independently.

Results:

```text
ROC-AUC : 0.9196
PR-AUC  : 0.0876
```

The key insight:

> **Anomaly does not necessarily mean fraud.**

The anomaly detector identified unusual transactions, but its fraud-ranking performance was far below the supervised XGBoost model.

---

# 🔗 8. Ensemble Evaluation

The project also evaluated whether combining:

```text
XGBoost + Isolation Forest
```

could improve performance.

The best validation configuration assigned:

```text
XGBoost Weight        : 1.0
Isolation Forest      : 0.0
```

PR-AUC remained:

```text
0.814246
```

Therefore, the anomaly detector was not artificially included simply to make the architecture look more sophisticated.

---

# 🧠 9. Explainability & Error Analysis

The pipeline goes beyond aggregate metrics.

Validation errors were explicitly inspected:

```text
Correct Predictions : 28,358
False Negatives     : 12
False Positives     : 3
```

Some missed fraud transactions received extremely low predicted fraud probabilities.

SHAP was used to investigate:

* Feature contribution
* Individual predictions
* Global model behavior
* False-positive / false-negative cases

The goal was not simply:

> “What is the score?”

but:

> **“Why did the model make this decision?”**

---

# ⚙️ 10. Production Inference

The trained model was converted into a reusable inference pipeline.

```text
Raw Transaction
       │
       ▼
Input Validation
       │
       ▼
Feature Engineering
       │
       ▼
XGBoost Model
       │
       ▼
Fraud Probability
       │
       ▼
Decision Threshold
       │
       ├───────────────┐
       ▼               ▼
    Genuine          Fraud
```

Saved artifacts include:

```text
fraud_xgboost_model.pkl
fraud_detector.pkl
```

This ensures that inference uses the same transformation logic developed during training.

---

# 📈 11. Model Drift Monitoring

A production model can degrade when the underlying data distribution changes.

The project therefore added **KS-based feature drift monitoring**.

Features with the largest observed KS statistics included:

```text
V21
V19
V11
Amount
V28
V23
V3
V13
V25
V1
```

The observed KS values were small, so this was treated as **potential distributional shift requiring monitoring**, not as evidence of major drift.

Generated report:

```text
stage15_drift_report.csv
```

---

# 🏆 Final Held-Out Test Results

The final evaluation was performed on the untouched test set.

| Metric        |     Result |
| ------------- | ---------: |
| **PR-AUC**    | **0.8476** |
| **ROC-AUC**   | **0.9859** |
| **Precision** |  **95.0%** |
| **Recall**    | **80.85%** |
| **F1 Score**  | **87.36%** |
| Accuracy      |     99.96% |

### Confusion Matrix

```text
                    Predicted
                 Genuine    Fraud
              ┌──────────┬─────────┐
Actual Genuine│  28,324  │    2    │
              ├──────────┼─────────┤
Actual Fraud  │     9    │   38    │
              └──────────┴─────────┘
```

**38 / 47 fraud transactions detected**

with

**2 false positives**

> Note: the test set contains only 47 fraud examples, so these metrics should be interpreted with appropriate uncertainty rather than treated as a universal production benchmark.

---

# 🧩 Complete ML Lifecycle

```text
┌────────────────────┐
│   DATA AUDITING    │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│  PREPROCESSING     │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ BASELINE MODELS    │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ IMBALANCE HANDLING │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ FEATURE ENGINEERING│
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ THRESHOLD OPTIM.   │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ HYPERPARAMETER OPT.│
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ ANOMALY DETECTION  │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ EXPLAINABILITY     │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ PRODUCTION INFER.  │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ DRIFT MONITORING   │
└────────────────────┘
```

---

# 🛠️ Tech Stack

| Category          | Technologies          |
| ----------------- | --------------------- |
| Language          | Python                |
| Data              | Pandas, NumPy         |
| ML                | Scikit-learn, XGBoost |
| Optimization      | Optuna, TPE           |
| Explainability    | SHAP                  |
| Anomaly Detection | Isolation Forest      |
| Visualization     | Matplotlib            |
| Model Persistence | Joblib / Pickle       |
| Monitoring        | KS Statistic          |

---

# 📁 Project Structure

```text
fraud-detection/
│
├── notebooks/
│   ├── 01_problem_understanding.ipynb
│   ├── 02_data_audit.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_baseline.ipynb
│   ├── 05_classical_ml.ipynb
│   ├── 06_imbalance.ipynb
│   ├── 07_threshold_optimization.ipynb
│   ├── 08_cost_sensitive.ipynb
│   ├── 09_feature_engineering.ipynb
│   ├── 10_hyperparameter_optimization.ipynb
│   ├── 11_anomaly_detection.ipynb
│   ├── 12_ensemble.ipynb
│   ├── 13_explainability.ipynb
│   ├── 14_production_inference.ipynb
│   └── 15_drift_monitoring.ipynb
│
├── models/
│   ├── fraud_xgboost_model.pkl
│   └── fraud_detector.pkl
│
├── reports/
│   └── stage15_drift_report.csv
│
├── requirements.txt
└── README.md
```

---

# 🎯 Key Engineering Takeaways

### 1. Accuracy is not enough

With extreme class imbalance, **PR-AUC, precision, recall and business costs** matter much more.

### 2. Threshold is a model decision layer

The model probability and the final business decision are two different things.

### 3. More complexity isn't automatically better

Isolation Forest and ensemble experiments did not improve the supervised model on this validation setup.

### 4. Hyperparameter optimization should have an objective

Optimization was performed against **PR-AUC**, rather than a generic metric.

### 5. Explainability matters

A production model should help answer **why** a prediction was made.

### 6. Deployment is not the end

Once deployed, the model needs **monitoring for distribution changes**.

---

# 🚀 Final Perspective

This project represents a shift from:

```text
Train a Model
```

to:

```text
Engineer an ML System
```

A real ML pipeline is not simply:

**Data → Model → Metric**

It is:

**Data → Validation → Features → Model → Decision → Explainability → Inference → Monitoring**

> **Build ML systems, not just ML models.**

---

## 👨💻 Author

**Saumya Srivastava**

B.Tech Information Technology | AI/ML Engineer

Interested in:

`Machine Learning` `Computer Vision` `LLMs` `RAG` `MLOps` `AI Engineering`

---

## ⭐ If You Found This Useful

Feel free to explore the implementation, experiment with the pipeline, and extend it with:

* Temporal validation
* Cross-validation
* Probability calibration
* Real-time streaming inference
* Feature stores
* Automated model retraining
* Production dashboards
* CI/CD for ML

**The goal is not to build a better notebook.
The goal is to build a better ML system.**
