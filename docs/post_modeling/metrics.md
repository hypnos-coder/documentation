# <span style="color: blue;">Metrics</span>


## Metrics Used for Model Evaluation

Our model evaluation revolves around the following key metrics:

- **AUC (Area Under the ROC Curve)**: Integrated into CatBoost training to assess the model's discriminative ability effectively.

- **Cross-Validated AUC**: Utilized during Hyperopt hyperparameter tuning to ensure robust performance estimation across multiple folds.

- **Learning Curve Analysis**: Employed to fine-tune parameters and resolve compatibility issues with specific bootstrap configurations.

These metrics collectively gauge model accuracy, robustness, and generalization capabilities.

Refer to detailed documentation for insights into metric implementation and significance in our model evaluation.

## Hyperparameter Tuning

### Hyperparameter Tuning Strategy

Our approach to hyperparameter tuning emphasizes maximizing AUC:

- **Initial CatBoost Training**: Focus on optimizing AUC during initial model training.

- **Hyperopt with Cross-Validation**: Refinement using Hyperopt and cross-validated AUC to select robust parameters and address bootstrap compatibility issues.

Explore our hyperparameter optimization methodologies in-depth through our dedicated documentation.
