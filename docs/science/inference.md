# Saving a CatBoost Model

In machine learning projects, saving trained models is essential for deployment and reproducibility. Here's how you can save a CatBoost model using the Joblib library:

```python
import joblib
from catboost import CatBoostClassifier

# ModelCB is our trained CatBoost model

model_filename = 'modelCB_collinear_true.joblib'
joblib.dump(modelCB, model_filename)

print(f"Model saved to {model_filename}")


# Loading a CatBoost Model

#Once a CatBoost model is trained and saved, . Here's how to load a saved CatBoost model using Joblib:


import joblib
from catboost import CatBoostClassifier

# Load the saved CatBoost model
model_filename = 'modelCB_collinear_true.joblib'
cat_loaded = joblib.load(model_filename)

print("Model loaded successfully")
```
# Progressive Feature Selection and Model Training with CatBoost

In this section, we demonstrate the process of progressively selecting the top features based on SHAP importances and training CatBoost models to find the best performing model.

### Setup

Ensure you have the necessary libraries installed:

```bash
pip install catboost shap
```
```python
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def train_and_store_models(X, y, shap_importances_df, feature_counts):
    models = {}
    auc_scores = {}

    for num_features_to_select in feature_counts:
        # Select the most important features
        top_features = shap_importances_df.head(num_features_to_select).index
        X_selected = X[top_features]
        
        # Indices of categorical features in the selected data
        categorical_features_indices = [X_selected.columns.get_loc(col) for col in X_selected.select_dtypes(include=['object', 'category']).columns]
        
        # Create CatBoost data pools
        train_pool = Pool(data=X_selected, label=y, cat_features=categorical_features_indices)
        valid_pool = Pool(data=X_selected, label=y, cat_features=categorical_features_indices)
        
        # Initialize CatBoost model with predefined parameters
        modelCB = CatBoostClassifier(
            iterations=6000,
            learning_rate=0.0036001849216860215,
            task_type='CPU',
            depth=11,
            l2_leaf_reg=0.00014951883662445424,
            bootstrap_type='Bernoulli',
            random_strength=0.006531581162153193,
            subsample=0.7730184738844886,
            loss_function='Logloss',
            eval_metric='AUC',
            used_ram_limit='3gb',
            random_seed=1,
            verbose=True,
            early_stopping_rounds=400,
            border_count=123,
            grow_policy='SymmetricTree',
            leaf_estimation_iterations=9,
            min_data_in_leaf=13
        )

        # Train the CatBoost model
        modelCB.fit(train_pool, eval_set=valid_pool, plot=True)
        
        # Store the model
        model_name = f'model_{num_features_to_select}_features'
        models[model_name] = modelCB
        
        # Predictions and calculate AUC
        y_pred_proba = modelCB.predict_proba(X_selected)[:, 1]
        auc = roc_auc_score(y, y_pred_proba)
        auc_scores[model_name] = auc
        print(f"AUC for {model_name}: {auc}")
    
    # Find the model with the best AUC score
    best_model_name = max(auc_scores, key=auc_scores.get)
    best_model = models[best_model_name]
    best_auc = auc_scores[best_model_name]
    
    print(f"Best model: {best_model_name} with AUC: {best_auc}")
    
    return models, auc_scores, best_model, best_model_name

# Define the list of feature counts to iterate over
feature_counts = [150, 120, 100, 85, 70, 75]

# Call the function to train and evaluate models
models, auc_scores, best_model, best_model_name = train_and_store_models(X, y, shap_importances_df, feature_counts)
import joblib

# Define the filename for saving the best model
model_filename = 'catboost_model_coli_150_.joblib'

# Save the best model using joblib
joblib.dump(best_model, model_filename)
print(f"Model saved to {model_filename}")

# Select the top features based on the best model's feature importance
num_features_to_select = 150
top_features = shap_importances_df.head(num_features_to_select).index
X_selected = X[top_features]
```
# Explaining CatBoost Model with SHAP

SHAP (SHapley Additive exPlanations) is a powerful tool for interpreting machine learning models. It provides insights into how individual features contribute to predictions made by the model. Here's how you can use SHAP to explain a CatBoost model:

### Setup



```bash
pip install shap
```
``` python
import joblib

# Load the saved CatBoost model
model_filename = 'modelCB_collinear_true.joblib'
cat_loaded = joblib.load(model_filename)

import shap

# Assuming X is our input data
explainer = shap.TreeExplainer(cat_loaded)
shap_values = explainer.shap_values(X)
#Visualize the summary plot of SHAP values to understand the overall feature importances:
shap.summary_plot(shap_values, X)
#Dependence Plots
shap.dependence_plot("Connexion", shap_values, X)
shap.dependence_plot("TAILLE_MENAGE", shap_values, X)
#Visualize the SHAP values for a specific prediction:
shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])
#Calculate and display the mean absolute SHAP importances across all features:
shap_importances = np.abs(shap_values).mean(axis=0)
shap_importances_df = pd.DataFrame(shap_importances, index=X.columns, columns=['Importance'])
shap_importances_df = shap_importances_df.sort_values(by='Importance', ascending=False)

print(shap_importances_df.head(10))
               Importance
Connexion        7.143799
TAILLE_MENAGE    0.354005
TypeLogmt_1      0.334522
 .1071           0.325862
H18E             0.253509
TypeLogmt_3      0.245747
TypeLogmt_2      0.241241
H09_Impute       0.209757
H08_Impute       0.203649
H20E             0.202173

```