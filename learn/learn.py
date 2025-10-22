import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import importlib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

DATA_FILE = "preprocessed_crypto_data.npz"
ARTIFACTS_DIR = "artifacts"
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
RANDOM_STATE = 42

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def get_svm_model():
    try:
        cuml = importlib.import_module('cuml')
        print("GPU cuML")
        return cuml.svm.SVC(probability=True, random_state=RANDOM_STATE)
    except ImportError:
        from sklearn.svm import SVC
        print(" sklearn SVC")
        return SVC(probability=True, random_state=RANDOM_STATE)

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"File not found {DATA_FILE}.")
    data = np.load(DATA_FILE)
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    return X_train_full, X_test, y_train_full, y_test

def evaluate_model(model, X_test, y_test, model_name: str):
    y_pred = model.predict(X_test)
    y_pred_proba = None
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n🔹 {model_name} result:")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    if y_pred_proba is not None:
        auc = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        print(f"ROC-AUC: {auc:.4f}")

        plt.figure()
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.title(f'ROC: {model_name}')

        plot_path = os.path.join(PLOTS_DIR, f"roc_{model_name}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"ROC-chart save: {plot_path}")

def train_with_gridsearch(model, params, X_train, y_train, name: str):
    print(f"\nModel learn{name} with GridSearchCV...")
    grid = GridSearchCV(model, params, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"Best params for {name}: {grid.best_params_}")
    print(f"Best F1-score: {grid.best_score_:.4f}")
    return grid.best_estimator_

def main():
    X_train, X_test, y_train, y_test = load_data()

    models = {
        'KNN': (KNeighborsClassifier(), {
            'n_neighbors': list(range(3, 21, 2)),
            'metric': ['euclidean', 'manhattan']
        }),
        'SVM': (get_svm_model(), {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }),
        'DecisionTree': (DecisionTreeClassifier(random_state=RANDOM_STATE), {
            'max_depth': [3, 5, 7, 10, None],
            'criterion': ['gini', 'entropy']
        }),
        'RandomForest': (RandomForestClassifier(random_state=RANDOM_STATE), {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_leaf': [1, 3, 5]
        })
    }

    for name, (model, params) in models.items():
        best_model = train_with_gridsearch(model, params, X_train, y_train, name)
        evaluate_model(best_model, X_test, y_test, name)

        if name == 'DecisionTree':
            plt.figure(figsize=(12, 8))
            plot_tree(best_model, filled=True, feature_names=['open', 'high', 'low', 'close', 'volume'], class_names=['Down', 'Up'])
            plt.title('Визуализация дерева решений')
            tree_path = os.path.join(PLOTS_DIR, "decision_tree.png")
            plt.savefig(tree_path)
            plt.close()

        if name == 'RandomForest':
            importances = best_model.feature_importances_
            plt.figure()
            plt.bar(['open', 'high', 'low', 'close', 'volume'], importances)
            plt.title('Feature Importances (Random Forest)')
            fi_path = os.path.join(PLOTS_DIR, "feature_importances.png")
            plt.savefig(fi_path)
            plt.close()

        model_path = os.path.join(MODELS_DIR, f"{name}_best.joblib")
        joblib.dump(best_model, model_path)
        print(f"Best model {name} was saved in: {model_path}")

if __name__ == "__main__":
    main()