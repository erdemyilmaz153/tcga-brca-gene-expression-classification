'''
TCGA BRCA Multi-Omics Tumor Type Classification

Dataset:
TCGA BRCA Multi-Omics

Objective:
Classify breast cancer histological tumor type using multi-omics molecular features.

Models:
- Logistic Regression (L1)
- Logistic Regression (L2)
- Random Forest

Evaluation:
- Stratified, shuffled 5-fold CV
- Accuracy
- Per-class classification reports
- Unweighted and weighted variants of each model

Results:
- All three models have higher cross-validation mean accuracy than majority baseline (~0.81).
- Within the models, cross-validation mean accuracy scores for L1, L2, and RF are ~0.91, ~0.90, and ~0.87, respectively.
- L1 has higher accuracy scores than Random Forest in each fold, and L2 has higher than Random Forest in four of the
folds while they are equal in one. However, in four folds, L1 has higher scores whereas, in one fold, L2 is higher.
Thus, even though logistic regression models have better performances than Random Forest, the separation between L1 and
L2 is not that clear.
- In class-weighted L1 and L2, weighting brought lobular precision and recall closer together, L1 and L2 remain
inseparable. For these models, weighting found about 8 more lobular tumors at the cost of about 8 more false alarms, so
accuracy stayed the same.
- Although accuracy for random forest increased by weighting, it is still lower than logistic regression models; also,
f1 scores for both classes are lower than the scores in logistic regression models. Additionally, accuracy understated
random forest's weakness; unweighted, it found only 37% of lobular tumors (vs. ~70% for logistic regression), despite
being lower by only ~0.03 in accuracy.

Interpretation:
After investigating the unweighted L1 model, it is found that while mutation in the gene CDH1 is the strongest feature
to classify as infiltrating lobular carcinoma; higher E-cadherin protein levels are the strongest feature to classify
as infiltrating ductal carcinoma.
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
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# 1. Problem definition

# Classifying breast cancer histological tumor type using multi-omics molecular features from the TCGA BRCA dataset.


# 2. Data loading

df = pd.read_csv('brca_data_w_subtypes.csv')


# 3. Basic exploration

print(df.shape)  # (705, 1941)

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

# The dataset is moderately imbalanced (81.4% vs 18.6%).

majority_baseline = y.value_counts(normalize=True).max()
print(f'Majority-baseline: {majority_baseline:.2f}')

X = df.drop(columns=[
    'vital.status',
    'PR.Status',
    'ER.Status',
    'HER2.Final.Status',
    'histological.type'
])


# 5. Modeling

# Logistic Regression with L2 Regularization (default)
pipeline_l2 = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(l1_ratio=0, max_iter=1000))
])

# Logistic Regression with L1 Regularization
pipeline_l1 = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=1000))
])

# Random Forest
rf_model = RandomForestClassifier(random_state=42)

# Logistic Regression models linear relationships, while Random Forest can capture non-linear patterns.


# 6. Evaluation

# 6.1. Cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
scores_l1 = cross_val_score(pipeline_l1, X, y, cv=cv)
scores_l2 = cross_val_score(pipeline_l2, X, y, cv=cv)
scores_rf = cross_val_score(rf_model, X, y, cv=cv)

# Standard deviations for the folds
print('Standard deviations for the folds:')
print(f'Logistic Regression (L1): {scores_l1.std():.3f}')  # 0.016
print(f'Logistic Regression (L2): {scores_l2.std():.3f}')  # 0.020
print(f'Random Forest Classifier: {scores_rf.std():.3f}')  # 0.018

# Accuracy difference between the folds
print('Accuracy difference between the folds:')

print(f'L1 and RF: {(scores_l1 - scores_rf).round(3)}')
# [0.035 0.035 0.021 0.035 0.05 ]
print(f'L2 and RF: {(scores_l2 - scores_rf).round(3)}')
# [0.    0.028 0.028 0.035 0.057]
print(f'L1 and L2: {(scores_l1 - scores_l2).round(3)}')
# [ 0.035  0.007 -0.007  0.    -0.007]

# Table of results
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
    ],
    "CV Accuracy Std": [
        scores_l1.std(),
        scores_l2.std(),
        scores_rf.std()
    ]
}).round(3)

results = results.sort_values("Mean CV Accuracy", ascending=False)
print(results)

results.plot(
    x="Model",
    y="Mean CV Accuracy",
    kind="bar"
)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

# Cross-validation is used to estimate model performance across different data splits.


# 6.2. Classification Reports

# 6.2.1. Logistic regression (L1)
y_pred_l1 = cross_val_predict(pipeline_l1, X, y, cv=cv)
default_l1_cr = classification_report(y, y_pred_l1, digits=3)
print(default_l1_cr)

# 6.2.2. Logistic regression (L2)
y_pred_l2 = cross_val_predict(pipeline_l2, X, y, cv=cv)
default_l2_cr = classification_report(y, y_pred_l2, digits=3)
print(default_l2_cr)

# 6.2.3. Random Forest
y_pred_rf = cross_val_predict(rf_model, X, y, cv=cv)
default_rf_cr = classification_report(y, y_pred_rf, digits=3)
print(default_rf_cr)


# 7. Class-weighted models

# 7.1. Logistic regression (L1)
l1_weighted = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        l1_ratio=1,
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000))
])

y_pred_l1_weighted = cross_val_predict(l1_weighted, X, y, cv=cv)
weighted_l1_cr = classification_report(y, y_pred_l1_weighted, digits=3)
print(weighted_l1_cr)

# 7.2. Logistic regression (L2)
l2_weighted = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        l1_ratio=0,
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    ))
])

y_pred_l2_weighted = cross_val_predict(l2_weighted, X, y, cv=cv)
weighted_l2_cr = classification_report(y, y_pred_l2_weighted, digits=3)
print(weighted_l2_cr)

# 7.3. Random Forest
rf_weighted = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)

y_pred_rf_weighted = cross_val_predict(rf_weighted, X, y, cv=cv)
weighted_rf_cr = classification_report(y, y_pred_rf_weighted, digits=3)
print(weighted_rf_cr)


# 8. Inspecting selected features by Logistic Regression (L1)

pipeline_l1.fit(X, y)

# 8.1. Classes
classes = pipeline_l1.named_steps['model'].classes_
print(f'The classes are: \n'
      f'Negative: {classes[0]}\n'
      f'Positive: {classes[1]}')

# 8.2. Coefficient strengths
coefficients = pipeline_l1.named_steps['model'].coef_[0]
feature_names = X.columns
coefs = pd.Series(coefficients, index=feature_names)

# 8.2.1. Towards infiltrating lobular carcinoma
print(f'Top five features pushing toward infiltrating lobular carcinoma: \n{coefs.sort_values(ascending=False).head()}')

# 8.2.2. Towards infiltrating ductal carcinoma
print(f'Top five features pushing toward infiltrating ductal carcinoma: \n{coefs.sort_values().head()}')

# 8.2.3. Number of features kept by the model
features_kept = (coefs != 0).sum()
print(f'Number of features kept by the model: {features_kept} ({(features_kept / len(coefs) * 100):.2f}%)')


# 9. Conclusion

# All three models have higher cross-validation mean accuracy score than majority baseline (0.81).

# For the models, L1 has higher accuracy scores than random forest in all folds; similarly, L2 has higher scores in
# four, while same in one. Thus, L1 and L2 slightly outperform random forest; the separation between L1 and L2 is not
# that clear.

# Although cross-validation mean accuracies for the models are higher than majority baseline, a detailed evaluation
# to observe minority-class performance is needed due to imbalance in the target variable. Before using weighted models,
# f1 score for ductal carcinoma of both L1 and L2 models is around 0.94 while it is around 0.93 for random forest. Since
# the target is imbalanced, it is more important to have more balanced (in terms of precision and recall) and higher f1
# score to be able to classify minority class successfully. In this case, even though precision scores are almost the
# same (0.79) for minority class in L1 and L2 models, L1 model has higher recall score (thus, detecting true lobular
# cases more successfully), and higher f1 score (0.74). For random forest, even though it has higher precision for
# lobular carcinoma, it has far lower recall (0.374) for it and so, poor f1 score with 0.524.

# Subsequently, balanced class-weights were used for the models. For L1 and L2 models, accuracy scores didn't change at
# all but slight increase in precision and a slight decrease in recall for ductal carcinoma class left their F1 scores
# almost unchanged. However, f1 score for lobular carcinoma in both models slightly increased: 0.742 to 0.758 in L1;
# 0.717 to 0.734 for L2. For L1, lower precision and higher recall values for minority class were observed; after the
# changes, they were closer to each other and so create more balanced model. In the case of random forest, accuracy
# increased from 0.874 to 0.895 and large increase for recall (0.37 to 0.62) and smaller decrease in precision (0.87 to
# 0.77) were seen. Even after these improvements in random forest, it is still behind for both logistic regression
# models. Therefore, although L1 and L2 models have almost the same accuracy scores, the weighted L1 model provides the
# most balanced approach to classifying tumors.

# After investigating the features selected by unweighted L1 regularization, it was seen that mutation in CDH1 gene has
# the highest positive coefficient which means it is the primary factor for the model to lean towards classifying the
# sample as infiltrating lobular carcinoma. On the other hand, higher E.Cadherin protein levels have the most negative
# coefficient which means it is the primary factor for the model to lean towards classifying the sample as infiltrating
# ductal carcinoma. The association between CDH1 mutation and lobular carcinoma is well established.

# It is known that the gene CDH1 is responsible for E.Cadherin production, and mutation in it is associated with
# infiltrating lobular carcinoma, the model successfully catches this feature and uses it as the primary factor in
# classification of the tumor types.


# 10. Limitations

# 1. Possible label noise: pathologists do not always agree on histological type, and some tumors have mixed ductal and
# lobular features.
# 2. Interpretation of selected features comes from a single fit on all data, and with correlated features a refit
# could select a different set.
# 3. No external validation dataset
# 4. This is a single cross-validation run. Small differences, like L1 vs. L2 ~8-tumor shifts, could change with a
# different split.
# 5. No costs were defined for the two kinds of error. Whether weighting counts as an improvement depends on costs.