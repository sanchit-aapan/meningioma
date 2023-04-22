# Meningioma ML Project - Sanchit Aapan

# Import relevant libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns


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


# Create a dictionary of the tuned classifiers
classifiers = {
    "LR": LogisticRegression(C=0.2, class_weight={0:1, 1:ratio}, fit_intercept=False, max_iter=200, penalty='l1', random_state=42, solver='liblinear', warm_start=False),
    "SVC": SVC(C=3, class_weight={0: 1, 1: ratio}, kernel='linear', probability=False, shrinking=False, random_state=42, tol=0.001, max_iter=500),
    "GaussianNB": GaussianNB(priors=[0.45, 0.55], var_smoothing=1e-9),
    "Ridge": RidgeClassifier(alpha=0.05, class_weight={0:1, 1:ratio}, copy_X=True, fit_intercept=False, max_iter=200, random_state=42, solver='lsqr', tol=0.001)
}


# Create a function to plot confusion matrices
def plot_conf_matrix(conf_matrix, name):

    plt.figure(figsize=(5,5))
    ax = sns.heatmap(conf_matrix, annot=True, cmap="Oranges" ,fmt='g', annot_kws={"size": 30}, cbar=False)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
    ax.set_title(f'{name}\n', fontsize=20)
    plt.show()


# Crate a function to print evaluation metrics
def evaluate(name, model):

    print(name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    ppv = precision_score(y_val, y_pred)
    npv = tn / (tn + fn)
    sensitivity = recall_score(y_val, y_pred)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    conf = confusion_matrix(y_val, y_pred)

    print(confusion_matrix(y_val, y_pred))
    print('PPV (precision):', ppv)
    print('NPV:', npv)
    print('Sensitivity (recall):', sensitivity)
    print('Specificity:', specificity)
    print("Accuracy:", accuracy)
    print("F1:", f1, '\n')

    plot_conf_matrix(conf, name)


# Create a function to plot ROC curves
def plot_roc():
    roc_curves = {}

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        try:
            y_score = clf.decision_function(X_val)
        except AttributeError:
            y_score = clf.predict_proba(X_val)[:, 1]
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


# Create a main function to run evaluation for each of the models
def run_evaluation():
    for clf in classifiers.items():
        evaluate(clf[0], clf[1])
    plot_roc()


# Run evaluation
run_evaluation()
