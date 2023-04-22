# Import relevant libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt
import numpy as np


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


# Create an instance of the winning model, a Ridge classifier, and fit it to the training data
ridge_clf = RidgeClassifier(alpha=0.05, class_weight={0:1, 1:ratio}, copy_X=True, fit_intercept=False, max_iter=200, random_state=42, solver='lsqr', tol=0.001)
# copy_X is set to True to avoid data overwriting
# tolerance is set to 0.001 to reduce computing time
# random state is set to make results reproducible
# other parameters are optimal values from the hyperparameter tuning process
ridge_clf.fit(X_train, y_train)


# Get feature importance as coefficients
coefficients = np.mean(ridge_clf.coef_, axis=0)


# Sort the coefficients according to their magnitudes, and sort their respective labels
sorted_idx = np.argsort(np.abs(coefficients))[::-1]
sorted_coefficients = coefficients[sorted_idx]
sorted_feature_names = [X.columns[i] for i in sorted_idx]

# Print sorted coefficients and feature names
print(sorted_coefficients)
print(sorted_feature_names)


# Show a graph of coefficients
plt.figure(figsize=(8, 4))
plt.barh(range(X.shape[1]), sorted_coefficients, tick_label=sorted_feature_names, color='#009193')  # Specifying data for the graph and colour of bars
plt.gca().invert_yaxis() # to make feature with the largest coefficient appear at the top
plt.title("Coefficients from Ridge Classifier")
plt.ylabel("Features")
plt.xlabel("Coefficient Value")
plt.show()