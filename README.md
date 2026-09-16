# Heart Disease Risk Prediction

Predicting 10-year risk of coronary heart disease (CHD) using the Framingham Heart Study dataset. This project covers the full early pipeline of a machine learning project: data cleaning, feature engineering, feature selection via regularization, and baseline classification modeling.

## Dataset

- **Source:** Framingham Heart Study (`heart_failure.csv`)
- **Size:** ~4,240 patients
- **Target:** `TenYearCHD` — whether the patient developed coronary heart disease within 10 years (1 = yes, 0 = no)
- **Predictors:** Clinical and lifestyle measurements — sex, age, education, smoking status, cigarettes/day, blood pressure medication, history of stroke/hypertension/diabetes, total cholesterol, systolic/diastolic blood pressure, BMI, heart rate, glucose

## What's in the notebook

1. **Data Overview** — shape, dtypes, summary statistics, category counts, and a data dictionary describing each column.
2. **Cleaning & Feature Engineering** — missing-value imputation (median for skewed continuous columns, mode for categorical), one-hot encoding for `education`, and two engineered features (smoking intensity category, heart rate category).
3. **Feature Selection via Regularization** — L1 (Lasso) and L2 (Ridge) penalized logistic regression fit across a range of regularization strengths, with a plot showing which features get driven to zero first.
4. **Classification & Model Evaluation** — train/test split, and Logistic Regression, LDA, and QDA fit and compared via 5-fold cross-validation on accuracy, precision, recall, and ROC-AUC, with ROC curves plotted for all three models.

## Tools

- Python, pandas, NumPy
- scikit-learn (preprocessing, linear models, discriminant analysis, model selection, metrics)
- Matplotlib

## Note

`random_state` values in this public copy are set to a placeholder rather than the author's actual student ID.
