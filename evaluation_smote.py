# Import relevant libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


# Import data with selected features into a Pandas dataframe
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


# Create a SMOTE-oversampled training dataset
oversample = SMOTE(random_state=42)
X_smote, y_smote = oversample.fit_resample(X_train, y_train)


# Create a list of the two classifiers (remove class weight balancing from SMOTE instance)
classifiers = [
    RidgeClassifier(alpha=0.05, class_weight={0:1, 1:ratio}, copy_X=True, fit_intercept=False, max_iter=200, random_state=42, solver='lsqr', tol=0.001),
    RidgeClassifier(alpha=0.05, class_weight=None, copy_X=True, fit_intercept=False, max_iter=200, random_state=42, solver='lsqr', tol=0.001)
]


# Create a list of the two training datasets
datasets = [[X_train, y_train], [X_smote, y_smote]]


# Crate a function to print evaluation metrics, and display ROC curves to get accurate AUC-ROC values
def evaluate(name, model, X_values, y_values):

    print(name)

    model.fit(X_values, y_values)

    y_pred = model.predict(X_val)

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    ppv = precision_score(y_val, y_pred)
    npv = tn / (tn + fn)
    sensitivity = recall_score(y_val, y_pred)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    roc_curves = {}

    model.fit(X_values, y_values)

    y_score = model.decision_function(X_val)

    fpr, tpr, _ = roc_curve(y_val, y_score)
    roc_curves[name] = (fpr, tpr)

    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr) in roc_curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black", alpha=0.8)
    plt.legend(loc="lower right")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")

    plt.grid()
    plt.show()

    print('PPV (precision):', ppv)
    print('NPV:', npv)
    print('Sensitivity (recall):', sensitivity)
    print('Specificity:', specificity)
    print('Accuracy:', accuracy)
    print('F1:', f1)



# Create a main function to run evaluation for each of the models
def run_evaluation():
    evaluate('CWB Ridge', classifiers[0], datasets[0][0], datasets[0][1])
    evaluate('SO Ridge', classifiers[1], datasets[1][0], datasets[1][1])


# Run evaluation
run_evaluation()
