# Import relevant libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# Import data into a Pandas dataframe
data = pd.read_csv("meningioma_data.csv")


# Split data into input features (X) and target variable (y)
X = data.drop(['MRN', 'Comp'], axis=1)
y = data['Comp']


# Split data into a modelling set and validation set - stratified to maintain class imbalance
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# Create a list with seven models, baseline and class weight-balanced

models = [
    (LogisticRegression(class_weight='balanced', random_state=42, penalty='l2', solver='lbfgs'), 'CWB LR-L2'),
    (LogisticRegression(class_weight='balanced', random_state=42, penalty='l1', solver='liblinear'), 'CWB LR-L1'),
    (SVC(class_weight='balanced', random_state=42), 'CWB SVC'),
    (RandomForestClassifier(class_weight='balanced', random_state=42), 'CWB RF'),
    (GaussianNB(priors=[0.5, 0.5]), 'CWB Gaussian'),
    (RidgeClassifier(class_weight='balanced', random_state=42), 'CWB Ridge'),
    (DecisionTreeClassifier(class_weight='balanced', random_state=42), 'CWB DTC'),
    (LogisticRegression(random_state=42, penalty='l2', solver='lbfgs'), 'LR-L2'),
    (LogisticRegression(random_state=42, penalty='l1', solver='liblinear'), 'LR-L1'),
    (SVC(random_state=42), 'SVC'),
    (RandomForestClassifier(random_state=42), 'RF'),
    (GaussianNB(), 'Gaussian'),
    (RidgeClassifier(random_state=42), 'Ridge'),
    (DecisionTreeClassifier(random_state=42), 'DTC')
]


# Create an instance of a stratified 5-fold cross validation with 10 repeats
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)


print('Average F1 scores over 5-fold cross-validation:')


# Print average F1 score for each model over the RSKF cross-validation
for model in models:
    cv_f1 = cross_val_score(model[0], X_train, y_train, cv=rskf, scoring='f1')
    print(f'{model[1]}:', np.mean(cv_f1))

