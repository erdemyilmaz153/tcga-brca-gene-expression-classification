'''
Gene Expression-Based Tumor Classification

Dataset:
TCGA BRCA Multi-Omics

Objective:
Classify histological tumor type using gene expression data.

Models:
- Logistic Regression (L1)
- Logistic Regression (L2)
- Random Forest

Evaluation:
5-fold cross-validation

Best Result:
Logistic Regression (L1) achieved ~0.91 mean accuracy.
'''

import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# 1. Problem definition

# Classifying tumor type using gene expression data in BRCA Multi-Omics (TCGA) dataset.


# 2. Data loading

df = pd.read_csv('brca_data_w_subtypes.csv')


# 3. Basic exploration

df.shape  # (705, 1941)

df.describe()

# The dataset contains 705 samples and 1936 numerical features.


# 4. Preprocessing

y = df['histological.type']

y.value_counts()
# infiltrating ductal carcinoma     574
# infiltrating lobular carcinoma    131

y.value_counts(normalize=True)
# infiltrating ductal carcinoma     0.814184
# infiltrating lobular carcinoma    0.185816

X = df.drop(columns=[
    'vital.status',
    'PR.Status',
    'ER.Status',
    'HER2.Final.Status',
    'histological.type'
])

# There are two different histological breast cancer types as 'infiltrating ductal carcinoma' and
# 'infiltrating lobular carcinoma'. The dataset is moderately imbalanced (81.4% vs 18.6%).


# 5. Modeling

# Logistic Regression with L2 Regularization (default)
pipeline_l2 = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(penalty="l2", max_iter=1000))
])

# Logistic Regression with L1 Regularization
pipeline_l1 = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000))
])

# Random Forest
rf_model = RandomForestClassifier(random_state=42)

# Logistic Regression models linear relationships, while Random Forest can capture non-linear patterns.


# 6. Evaluation

scores_l1 = cross_val_score(pipeline_l1, X, y, cv=5)
scores_l2 = cross_val_score(pipeline_l2, X, y, cv=5)
scores_rf = cross_val_score(rf_model, X, y, cv=5)

results = pd.DataFrame({
    "Model": [
        "Logistic Regression (L1)",
        "Logistic Regression (L2)",
        "Random Forest"
    ],
    "Mean CV Accuracy": [
        scores_l1.mean(),
        scores_l2.mean(),
        scores_rf.mean()
    ]
})

results = results.sort_values("Mean CV Accuracy", ascending=False)
print(results)

results.plot(
    x="Model",
    y="Mean CV Accuracy",
    kind="bar"
)
plt.show()

# Cross-validation is used to estimate model performance across different data splits.


# 7. Conclusion

# Logistic Regression with L1 regularization achieved the highest mean cross-validation accuracy (~0.91).
# Random Forest achieved slightly lower performance (~0.87).


# 8. Limitations

# Dataset is relatively small
# High dimensionality (p >> n)
# Possible label noise
# No external validation dataset