# Import relevant libraries
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Import data into a Pandas dataframe
data = pd.read_csv("meningioma_data.csv")


# Split data into input features (X) and target variable (y)
X = data.drop(['MRN', 'Comp'], axis=1)
y = data['Comp']


# Split data into a modelling set and validation set - stratified to maintain class imbalance
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# Calculate class imbalance ratio from training data
count_class_1 = y_train.value_counts()[0]
count_class_2 = y_train.value_counts()[1]
ratio = count_class_1/count_class_2


# Create a list of models to tune
models = [
    LogisticRegression(),
    SVC(),
    GaussianNB(),
    RidgeClassifier(),
]


# Create one or more hyperparameter grid(s) for each model

# A list of three hyperparameter grids for logistic regression
lr_param_grid = [
    {
        'penalty': ['l1'],
        'C': np.logspace(-4, 4, 20),
        'fit_intercept': [True, False],
        'intercept_scaling': [0.1, 1, 10],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200, 500, 1000],
        'class_weight': [None, 'balanced', {0:1, 1:ratio}],
        'warm_start': [False],
        'random_state': [42]
    },
    {
        'penalty': ['l2'],
        'C': np.logspace(-4, 4, 20),
        'fit_intercept': [True, False],
        'intercept_scaling': [0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga', 'newton-cholesky'],
        'max_iter': [100, 200, 500, 1000],
        'class_weight': [None, 'balanced', {0:1, 1:ratio}],
        'warm_start': [False],
        'random_state': [42]
    },
    {
        'penalty': ['elasticnet'],
        'C': np.logspace(-4, 4, 20),
        'fit_intercept': [True, False],
        'intercept_scaling': [0.1, 1, 10],
        'solver': ['saga'],
        'max_iter': [100, 200, 500, 1000],
        'class_weight': [None, 'balanced', {0:1, 1:ratio}],
        'warm_start': [False],
        'l1_ratio': np.linspace(0, 1, 10),
        'random_state': [42]
    }
]

# A hyperparameter grid for SVC
svc_param_grid = {
    'C': np.logspace(-2, 2, 10),
    'kernel': ['linear'],
    'shrinking': [False],
    'probability': [True, False],
    'class_weight': [None, 'balanced', {0:1, 1:ratio}],
    'random_state': [42],
    'max_iter': [100, 200, 500, 1000],
}


# A hyperparameter grid for GNB
gaussian_param_grid = {
    'priors': [(p, 1 - p) for p in np.linspace(0, 1, 30)],
    'var_smoothing': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
}


# A hyperparameter grid for Ridge
ridge_param_grid = {
        'alpha': [0.05, 0.1, 0.5, 1, 10],
        'fit_intercept': [True, False],
        'copy_X': [True],
        'max_iter': [None, 100, 500],
        'tol': [1e-3],
        'class_weight': [None, 'balanced', {0:1, 1:ratio}, {0:1, 1:4}, {0:1, 1:4.5}, {0:1, 1:5}, {0:1, 1:5.5}],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        'random_state': [42]
}


# Create a function to tune hyperparameters for each model with respective grids
def param_tune(model, grid):

    # Create an instance of a stratified 5-fold cross validation with 10 repeats
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    # Create a grid search optimising each model for F1 score using the RSKF cross-validation
    grid_search = GridSearchCV(model, grid, scoring='f1', cv=rskf, verbose=1, n_jobs=-1)

    # Fit the grid search to the training data and get the best scores
    grid_search.fit(X_train, y_train)

    best_f1_score = grid_search.best_score_

    # Find all iterations with the highest F1 score
    best_combinations = []
    for i, (params, mean_test_score) in enumerate(zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'])):
        if np.round(mean_test_score, 2) == np.round(best_f1_score, 2):
            best_combinations.append((i, params, mean_test_score))

    # Print the best combinations
    for idx, params, f1_score in best_combinations:
        print(f"Index: {idx}, Parameters: {params}, F1 Score: {f1_score:.4f}")


# Create a function to run the tuning process
def run_tuning():
    param_tune(models[0], lr_param_grid)
    param_tune(models[1], svc_param_grid)
    param_tune(models[2], gaussian_param_grid)
    param_tune(models[3], ridge_param_grid)


# Run the hyperparameter tuning process
run_tuning()


# A record of the selected parameters from the process, as a reference
lr_tuned = LogisticRegression(C=0.2, class_weight={0:1, 1:ratio}, fit_intercept=False, intercept_scaling=1, max_iter=200, penalty='l1', random_state=42, solver='liblinear', warm_start=False)
svc_tuned = SVC(C=3, class_weight={0: 1, 1: ratio}, kernel='linear', probability=False, shrinking=False, random_state=42, tol=0.001, max_iter=500)
gaussian_tuned = GaussianNB(priors=[0.5, 0.5], var_smoothing=1e-20)
ridge_tuned = RidgeClassifier(alpha=0.1, class_weight={0:1, 1:ratio}, copy_X=True, fit_intercept=False, max_iter=200, random_state=42, solver='lsqr', tol=0.001)
