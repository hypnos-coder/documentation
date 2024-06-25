# Hyperparameter Tuning with Hyperopt and CatBoost

In this section, we describe the process of hyperparameter tuning for the CatBoostClassifier using Hyperopt.

## Hyperparameter Space Definition

Define the search space for hyperparameters using Hyperopt's syntax. Here's an example:

```python
from catboost import CatBoostClassifier, CatboostError
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope
import numpy as np
from sklearn.metrics import roc_auc_score

# Define the hyperparameter search space
space = {
    'depth': hp.choice('depth', [2, 3, 4, 5, 6, 10]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.5)),
    'l2_leaf_reg': hp.loguniform('l2_leaf_reg', np.log(1e-8), np.log(10)),
    'random_strength': hp.loguniform('random_strength', np.log(1e-8), np.log(10)),
    'border_count': scope.int(hp.quniform('border_count', 1, 255, 1)),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'bootstrap_type': hp.choice('bootstrap_type', [
        {'type': 'Bayesian', 'bagging_temperature': hp.uniform('bagging_temperature', 0, 1)},
        {'type': 'Bernoulli'},
        {'type': 'Poisson'},
        {'type': 'MVS'}
    ]),
    'grow_policy': hp.choice('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
    'min_data_in_leaf': scope.int(hp.quniform('min_data_in_leaf', 1, 20, 1)),
    'leaf_estimation_iterations': scope.int(hp.quniform('leaf_estimation_iterations', 1, 10, 1))
}
# Define the objective function
def objective(params):
    # Extract bootstrap_type and adjust parameters accordingly
    bootstrap_type_params = params['bootstrap_type']
    if bootstrap_type_params['type'] == 'Bayesian':
        params['bagging_temperature'] = bootstrap_type_params['bagging_temperature']
    elif bootstrap_type_params['type'] in ['Bernoulli', 'Poisson', 'MVS']:
        params['subsample'] = params['subsample']
    params['bootstrap_type'] = bootstrap_type_params['type']

    # Remove unused parameters
    if params['bootstrap_type'] != 'Bayesian':
        params.pop('bagging_temperature', None)
    if params['bootstrap_type'] == 'Bayesian':
        params.pop('subsample', None)

    # Initialize CatBoost model with specified parameters
    model = CatBoostClassifier(
        iterations=4000,
        depth=params['depth'],
        learning_rate=params['learning_rate'],
        l2_leaf_reg=params['l2_leaf_reg'],
        random_strength=params['random_strength'],
        border_count=params['border_count'],
        subsample=params.get('subsample', None),
        bagging_temperature=params.get('bagging_temperature', None),
        bootstrap_type=params['bootstrap_type'],
        grow_policy=params['grow_policy'],
        min_data_in_leaf=params['min_data_in_leaf'],
        leaf_estimation_iterations=params['leaf_estimation_iterations'],
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=12345,
        verbose=False,
        early_stopping_rounds=150
    )

    # Train the model
    try:
        model.fit(train_pool, eval_set=valid_pool, verbose=False)
    except CatboostError as e:
        # Return a high loss value in case of error
        print(f"CatBoostError: {e}")
        return {'loss': 9999, 'status': STATUS_OK}

    # Predict and compute AUC score
    y_pred = model.predict_proba(X_valid_cat)[:, 1]
    auc = roc_auc_score(y_valid, y_pred)
    return {'loss': -auc, 'status': STATUS_OK}
# Perform the optimization
trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
            rstate=np.random.default_rng(12345))

# Display the best hyperparameters found
print("Best hyperparameters:", best)
```

### Model Adjustment and Cross-Validation

In this section, we adjust the CatBoostClassifier model with specific parameters and evaluate its performance using cross-validation.

## Adjusting the Model

First, adjust the CatBoostClassifier model with specific parameters and evaluate its initial performance.

```python
from catboost import CatBoostClassifier, cv, Pool
from sklearn.metrics import roc_auc_score

# Initialize CatBoostClassifier with specific hyperparameters
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
    save_snapshot=True,
    snapshot_file='catboost_snapshot_test_all_',
    early_stopping_rounds=400,
    border_count=123,
    grow_policy='SymmetricTree',
    leaf_estimation_iterations=9,
    min_data_in_leaf=13
)

# Fit the model to training data and evaluate performance
modelCB.fit(train_pool, eval_set=valid_pool, plot=True)
y_pred_probaCB = modelCB.predict_proba(X)[:, 1]
auc = roc_auc_score(y, y_pred_probaCB)
print("Initial AUC: ", auc)

# Define parameters for cross-validation
params = {
    'iterations': 6000,
    'learning_rate': 0.0036001849216860215,
    'task_type': 'CPU',
    'depth': 11,
    'l2_leaf_reg': 0.00014951883662445424,
    'bootstrap_type': 'Bernoulli',
    'random_strength': 0.006531581162153193,
    'subsample': 0.7730184738844886,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'used_ram_limit': '3gb',
    'random_seed': 1,
    'verbose': True,
    'save_snapshot': True,
    'snapshot_file': 'catboost_snapshot_test_all_',
    'early_stopping_rounds': 400,
    'border_count': 123,
    'grow_policy': 'SymmetricTree',
    'leaf_estimation_iterations': 9,
    'min_data_in_leaf': 13
}

# Perform cross-validation
cv_results = cv(
    pool=train_pool,
    params=params,
    fold_count=8,
    shuffle=True,
    partition_random_seed=42,
    plot=True  # Display cross-validation plot
)

# Extract the mean AUC from cross-validation results
mean_auc = cv_results['test-AUC-mean'].max()
print("Mean AUC from Cross-Validation: ", mean_auc)
```
## Training and Evaluation of Models with Various Techniques in a Pipeline

In this section, we explore the application of different machine learning models using a pipeline approach after feature selection with SHAP values. The pipeline includes preprocessing steps for numerical, categorical, and binary features, followed by training and evaluation using several classifiers.

### Preprocessing

We start by defining pipelines for numerical scaling and categorical encoding:

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Define transformers for preprocessing
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformer for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('bin', 'passthrough', binary_features)
    ])
#Model Initialization
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier

# Initialize models
lgb_model = LGBMClassifier(
    objective='binary',
    learning_rate=0.2,
    max_depth=30,
    num_leaves=51,
    metric='auc',
    boosting_type='gbdt',
    verbosity=1,
    seed=42,
    n_estimators=3000
)

clf_mlp = MLPClassifier(
    hidden_layer_sizes=(16, 16, 16, 16),
    activation='relu',
    solver='adam',
    alpha=0.1,
    batch_size=16,
    learning_rate='constant',
    learning_rate_init=0.0001,
    max_iter=1500,
    random_state=1,
    verbose=1,
    tol=0.00001,
    n_iter_no_change=50
)

params_xgb = {
    'alpha': 0.005980079696976115,
    'colsample_bytree': 0.6,
    'eta': 0.27123728532406893,
    'gamma': 0.2,
    'lambda': 0.5,
    'max_depth': 10,
    'min_child_weight': 2.0,
    'subsample': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'seed': 42
}
xgb_model = XGBClassifier(**params_xgb, use_label_encoder=False, verbosity=0)

rf_model = RandomForestClassifier(n_estimators=2000)

dt_model = DecisionTreeClassifier(criterion='gini', max_depth=None, max_features=None, min_samples_leaf=5, min_samples_split=15)

bagging_model = BaggingClassifier(estimator=dt_model, n_estimators=100, random_state=42)
#Stacking Classifier
# List of base models for stacking
estimators = [
    ('lgb_model', lgb_model),
    ('mlp', clf_mlp),
    ('xgboost', xgb_model),
    ('catboost', best_model),
    ('randomF', rf_model),
    ('decision_tree', bagging_model)
]

# Initialize StackingClassifier with a final estimator
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=GradientBoostingClassifier(
        random_state=1, learning_rate=0.01, loss='log_loss', n_estimators=300, verbose=1
    ),
    cv=5
)

# Pipeline including preprocessing and stacking
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('stacking', stacking)
])

# Fit the model on selected features
model.fit(X_selected, y)

```