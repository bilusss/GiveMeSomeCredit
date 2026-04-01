# GiveMeSomeCredit

Kaggle competition: predict if a client will default on debt (binary classification).

## Results

| Metric | Value |
|--------|-------|
| **Kaggle Private Score** | 0.86678 |
| **AUC (CV)** | 0.8656 ± 0.0036 |
| **GINI** | 0.7334 |

**Best model**: Gradient Boosting (n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8)

**Top 3 features**: TotalLatePayments, RevolvingUtilizationOfUnsecuredLines, NumberOfTimes90DaysLate

---

## Project Structure

```
GiveMeSomeCredit/
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb    # Data cleaning & feature engineering
│   ├── 03_modeling.ipynb         # Model training & hyperparameter tuning
│   └── 04_evaluation.ipynb       # Performance analysis & visualizations
├── src/
│   └── preprocessing.py          # Reusable preprocessing functions
├── data/
│   ├── raw/                      # Original Kaggle data
│   └── processed/                # Cleaned train/test CSVs
├── submission.csv                # Final Kaggle submission
└── requirements.txt
```

---

## Data Description

- **~250k examples** (150k train, 100k test)
- **7% positive class** (imbalanced - accuracy won't work, using AUC)
- **Missing values**: MonthlyIncome (19.82%), NumberOfDependents (2.62%)

### Variables

| Variable | Description | Type |
|----------|-------------|------|
| **SeriousDlqin2yrs** | Person experienced 90+ days past due delinquency (TARGET) | binary |
| RevolvingUtilizationOfUnsecuredLines | Credit utilization ratio | % |
| age | Age of borrower | int |
| NumberOfTime30-59DaysPastDueNotWorse | Times 30-59 days late (last 2 years) | int |
| DebtRatio | Monthly debt / gross income | % |
| MonthlyIncome | Monthly income | real |
| NumberOfOpenCreditLinesAndLoans | Open loans and credit lines | int |
| NumberOfTimes90DaysLate | Times 90+ days late | int |
| NumberRealEstateLoansOrLines | Mortgage/real estate loans | int |
| NumberOfTime60-89DaysPastDueNotWorse | Times 60-89 days late (last 2 years) | int |
| NumberOfDependents | Family dependents | int |

> Source: [Kaggle Competition](https://www.kaggle.com/competitions/GiveMeSomeCredit/overview)

---

## Approach

### 1. EDA (`01_eda.ipynb`)

- [x] Identified nulls in MonthlyIncome (19.82%) and NumberOfDependents (2.62%)
- [x] Confirmed class imbalance (93%/7%) - switched to AUC metric
- [x] Found data entry errors: age=0, error codes 96/98 in late payment columns
- [x] Correlation heatmap analysis

### 2. Preprocessing (`02_preprocessing.ipynb` + `src/preprocessing.py`)

- [x] Dropped rows where age < 18
- [x] Filled nulls with median per age group (bins: 0-30, 30-45, 45-60, 60-75, 75-120)
- [x] Avoided data leakage: medians computed on train set only
- [x] Feature engineering:
  - `MonthlyDebt` = DebtRatio × MonthlyIncome
  - `TotalLatePayments` = sum of all late payment columns
- [x] Removed index column, saved processed data

### 3. Modeling (`03_modeling.ipynb`)

- [x] Compared 4 models with StratifiedKFold CV:

| Model | AUC |
|-------|-----|
| Logistic Regression | 0.7905 |
| Random Forest | 0.8364 |
| XGBoost | 0.8490 |
| **Gradient Boosting** | **0.8646** |

- [x] Pipeline with StandardScaler (prevents data leakage)
- [x] Hyperparameter tuning with RandomizedSearchCV
- [x] Generated Kaggle submission

### 4. Evaluation (`04_evaluation.ipynb`)

- [x] Cross-validation performance analysis
- [x] Feature importance visualization
- [x] ROC curve and Precision-Recall curves
- [x] Prediction distribution by class

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebooks in order:
# 1. 01_eda.ipynb
# 2. 02_preprocessing.ipynb
# 3. 03_modeling.ipynb
# 4. 04_evaluation.ipynb
```

---

## Tech Stack

- Python 3.14
- pandas, numpy
- scikit-learn, xgboost
- matplotlib, seaborn
- Jupyter