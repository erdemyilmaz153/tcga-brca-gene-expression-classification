# TCGA BRCA Gene Expression Classification

## Objective
- Classify histological tumor type (infiltrating ductal carcinoma vs infiltrating lobular carcinoma) using gene expression data.

## Dataset
- TCGA BRCA Multi-Omics

## Models Used
- Logistic Regression (L1)
- Logistic Regression (L2)
- Random Forest

## Evaluation Method
- 5-fold cross-validation

## Results
| Model | Mean CV Accuracy |
|---------|---------:|
| Logistic Regression (L1) | 0.913 |
| Logistic Regression (L2) | 0.895 |
| Random Forest | 0.868 |

## Best Result
- Logistic Regression (L1) achieved the highest mean cross-validation accuracy (0.913).

## Limitations
- Dataset is relatively small
- High dimensionality (p >> n)
- Possible label noise
- No external validation dataset