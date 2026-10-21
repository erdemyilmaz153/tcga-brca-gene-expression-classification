# TCGA BRCA Multi-Omics Tumor Type Classification

## Objective
- Classify breast cancer histological tumor type using multi-omics molecular features from the TCGA BRCA dataset.

## Dataset
- TCGA BRCA Multi-Omics dataset

## Models Used
- Logistic Regression (L1)
- Logistic Regression (L2)
- Random Forest

## Evaluation
- Stratified, shuffled 5-fold CV
- Accuracy
- Per-class classification reports
- Unweighted and weighted variants of each model

## Results
| Model | Mean CV Accuracy | Lobular Carcinoma Precision | Lobular Carcinoma Recall | Lobular Carcinoma F-1 Score |
|---------|---------|---------|---------|---------:|
| Logistic Regression (L1) | 0.909 | 0.786 | 0.702 | 0.742 |
| Logistic Regression (L2) | 0.904 | 0.789 | 0.656 | 0.717 |
| Random Forest | 0.874 | 0.875 | 0.374 | 0.524 |
| Logistic Regression (L1), weighted | 0.909 | 0.752 | 0.763 | 0.758 |
| Logistic Regression (L2), weighted | 0.904 | 0.752 | 0.718 | 0.734 |
| Random Forest, weighted | 0.895 | 0.771 | 0.618 | 0.686 |

## Model Comparison
- All three models have higher cross-validation mean accuracy score than majority baseline (0.81).
- Weighted or not, Logistic Regression (L1) and Logistic Regression (L2) models have similar accuracy scores, and larger than the random forest model. 
- In weighted models, Logistic Regression (L1) has slightly higher f1-score for lobular carcinoma class; but it is a single split and needs to be considered carefully.

## Key Finding
- Weighting brought precision and recall closer together providing more balanced approach. For logistic regression models, about 8 more lobular tumors found and about 8 more false alarms, with accuracy unchanged.
- Accuracy understated Random Forest's weakness; for the unweighted model, it found only 37% of lobular tumors (vs. 70% for logistic regression), despite being lower by only ~0.03 in accuracy.

## Interpretation
- After investigating the unweighted L1 model, it is found that while mutation in the gene CDH1 is the strongest feature to classify as infiltrating lobular carcinoma; higher E-cadherin protein levels are the strongest feature to classify as infiltrating ductal carcinoma.
- CDH1 encodes E-cadherin: the gene is damaged in lobular tumors, and its protein is present in ductal ones, consistent with well-established biology.
 

## Limitations
1. Possible label noise: pathologists do not always agree on histological type, and some tumors have mixed ductal and lobular features.
2. Interpretation of selected features comes from a single fit on all data, and with correlated features a refit could select a different set.
3. No external validation dataset.
4. This is a single cross-validation run. Small differences, like L1 vs. L2 ~8-tumor shifts, could change with a different split.
5. No costs were defined for the two kinds of error. Whether weighting counts as an improvement depends on costs.

## Revision Notes

**October 2026.** The original version evaluated models on accuracy alone and reported L1 as the best model (~0.91). The revision added:

- **Majority-class baseline (0.81)**, so accuracy can be read in context.
- **Explicit stratified, shuffled 5-fold CV.** The model comparison held under the new split.
- **Fold-by-fold model comparison** instead of comparing means only. Logistic regression outperformed random forest in every fold, but L1 and L2 could not be separated, so the original "L1 is the best" claim was dropped.
- **Per-class classification reports**, which showed that accuracy hid random forest's weakness on the minority class (lobular recall 0.374).
- **Class-weighted variants of each model**, with the precision/recall trade-off reported.
- **Inspection of L1-selected features**, which showed the model relies on CDH1 / E-cadherin, consistent with known lobular carcinoma biology.
- **Specific limitations** in place of generic ones.
- **The deprecated `penalty` argument replaced with `l1_ratio`** (scikit-learn 1.8), after confirming the L1 model still selects the same 185 features.