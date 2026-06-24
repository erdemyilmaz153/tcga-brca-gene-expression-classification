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
- 5-fold Cross-Validation
- Accuracy

## Results
| Model | Mean CV Accuracy |
|---------|---------:|
| Logistic Regression (L1) | 0.913 |
| Logistic Regression (L2) | 0.895 |
| Random Forest | 0.868 |

## Best Result
- Logistic Regression (L1) achieved the highest mean cross-validation accuracy (0.913), suggesting that sparse feature selection was beneficial for this high-dimensional dataset.

## Interpretation
- Logistic Regression (L1) outperformed L2 regularization and Random Forest.
- L1 regularization likely benefited from the high-dimensional nature of the dataset by selecting a sparse subset of informative features.

## Limitations
- Dataset is relatively small
- High dimensionality (p >> n)
- Possible label noise
- No external validation dataset