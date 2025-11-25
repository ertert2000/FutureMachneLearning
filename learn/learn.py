import numpy as np
from tqdm import tqdm
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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
    from sklearn.svm import LinearSVC
    return LinearSVC(random_state=RANDOM_STATE)


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

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n🔹 {model_name} result:")
    print(classification_report(
        y_test, y_pred,
        digits=3,
        target_names=["Down (-1)", "Flat (0)", "Up (1)"]
    ))
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")



def train_with_gridsearch(model, params, X_train, y_train, name: str):
    print(f"\nModel learn {name} with GridSearchCV...")
    grid = GridSearchCV(model, params, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)
    print(f"Best params for {name}: {grid.best_params_}")
    print(f"Best F1-weighted score: {grid.best_score_:.4f}")
    return grid.best_estimator_


def main():
    X_train, X_test, y_train, y_test = load_data()

    models = {
        'KNN': (KNeighborsClassifier(), {
            'n_neighbors': list(range(3, 21, 2)),
            'metric': ['euclidean', 'manhattan']
        }),
        'DecisionTree': (DecisionTreeClassifier(random_state=RANDOM_STATE), {
            'max_depth': [3, 5, 7, 10, None],
            'criterion': ['gini', 'entropy']
        }),
        'RandomForest': (RandomForestClassifier(random_state=RANDOM_STATE), {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_leaf': [1, 3, 5]
        }),
        'SVM': (get_svm_model(), None)
    }

    for name, (model, params) in tqdm(models.items(), desc="Training models"):
        if params is not None:
            best_model = train_with_gridsearch(model, params, X_train, y_train, name)
        else:
            print(f"\nTraining {name} without GridSearch...")
            model.fit(X_train, y_train)
            best_model = model

        evaluate_model(best_model, X_test, y_test, name)

        if name == 'DecisionTree':
            plt.figure(figsize=(12, 8))
            plot_tree(
                best_model,
                filled=True,
                feature_names=['open', 'high', 'low', 'close', 'volume'],
                class_names=['Down', 'Flat', 'Up']
            )
            plt.title('Decision Tree')
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


# import numpy as np
# from tqdm import tqdm
# import joblib
# import os
# import matplotlib.pyplot as plt
# import importlib
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     roc_curve,
#     classification_report
# )
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.ensemble import RandomForestClassifier

# DATA_FILE = "preprocessed_crypto_data.npz"
# ARTIFACTS_DIR = "artifacts"
# PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")
# MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
# RANDOM_STATE = 42

# os.makedirs(PLOTS_DIR, exist_ok=True)
# os.makedirs(MODELS_DIR, exist_ok=True)

# def get_svm_model():
#     from sklearn.svm import LinearSVC
#     from sklearn.calibration import CalibratedClassifierCV
#     base_svm = LinearSVC(random_state=RANDOM_STATE)
#     calibrated_svm = CalibratedClassifierCV(base_svm, cv=5)
#     return calibrated_svm

# def load_data():
#     if not os.path.exists(DATA_FILE):
#         raise FileNotFoundError(f"File not found {DATA_FILE}.")
#     data = np.load(DATA_FILE)
#     X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
#     y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

#     X_train_full = np.vstack([X_train, X_val])
#     y_train_full = np.concatenate([y_train, y_val])

#     return X_train_full, X_test, y_train_full, y_test

# def evaluate_model(model, X_test, y_test, model_name: str):
#     y_pred = model.predict(X_test)
#     y_pred_proba = None
#     if hasattr(model, "predict_proba"):
#         y_pred_proba = model.predict_proba(X_test)[:, 1]

#     acc = accuracy_score(y_test, y_pred)
#     prec = precision_score(y_test, y_pred)
#     rec = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)

#     print(f"\n🔹 {model_name} result:")
#     print(classification_report(y_test, y_pred, digits=3))
#     print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

#     if y_pred_proba is not None:
#         auc = roc_auc_score(y_test, y_pred_proba)
#         fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
#         print(f"ROC-AUC: {auc:.4f}")

#         plt.figure()
#         plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
#         plt.plot([0, 1], [0, 1], 'k--')
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.legend()
#         plt.title(f'ROC: {model_name}')

#         plot_path = os.path.join(PLOTS_DIR, f"roc_{model_name}.png")
#         plt.savefig(plot_path)
#         plt.close()
#         print(f"ROC-chart save: {plot_path}")

# def train_with_gridsearch(model, params, X_train, y_train, name: str):
#     print(f"\nModel learn{name} with GridSearchCV...")
#     grid = GridSearchCV(model, params, cv=5, scoring='f1', n_jobs=-1, verbose=2)
#     grid.fit(X_train, y_train)
#     print(f"Best params for {name}: {grid.best_params_}")
#     print(f"Best F1-score: {grid.best_score_:.4f}")
#     return grid.best_estimator_

# def main():
#     X_train, X_test, y_train, y_test = load_data()

#     models = {
#         'KNN': (KNeighborsClassifier(), {
#             'n_neighbors': list(range(3, 21, 2)),
#             'metric': ['euclidean', 'manhattan']
#         }),
#         'DecisionTree': (DecisionTreeClassifier(random_state=RANDOM_STATE), {
#             'max_depth': [3, 5, 7, 10, None],
#             'criterion': ['gini', 'entropy']
#         }),
#         'RandomForest': (RandomForestClassifier(random_state=RANDOM_STATE), {
#             'n_estimators': [50, 100, 200],
#             'max_depth': [5, 10, None],
#             'min_samples_leaf': [1, 3, 5]
#         }),
#         'SVM': (get_svm_model(), None)
#     }

#     for name, (model, params) in tqdm(models.items(), desc="Training models"):
#         if params is not None:
#             best_model = train_with_gridsearch(model, params, X_train, y_train, name)
#         else:
#             print(f"\nTraining {name} without GridSearch...")
#             if name == 'SVM':
#                 model.set_params()
#             model.fit(X_train, y_train)
#             best_model = model

#         evaluate_model(best_model, X_test, y_test, name)

#         if name == 'DecisionTree':
#             plt.figure(figsize=(12, 8))
#             plot_tree(best_model, filled=True, feature_names=['open', 'high', 'low', 'close', 'volume'], class_names=['Down', 'Up'])
#             plt.title('Визуализация дерева решений')
#             tree_path = os.path.join(PLOTS_DIR, "decision_tree.png")
#             plt.savefig(tree_path)
#             plt.close()

#         if name == 'RandomForest':
#             importances = best_model.feature_importances_
#             plt.figure()
#             plt.bar(['open', 'high', 'low', 'close', 'volume'], importances)
#             plt.title('Feature Importances (Random Forest)')
#             fi_path = os.path.join(PLOTS_DIR, "feature_importances.png")
#             plt.savefig(fi_path)
#             plt.close()

#         model_path = os.path.join(MODELS_DIR, f"{name}_best.joblib")
#         joblib.dump(best_model, model_path)
#         print(f"Best model {name} was saved in: {model_path}")

# if __name__ == "__main__":
#     main()

# output
# Training models:   0%|                                                                                                                                                                          | 0/4 [00:00<?, ?it/s]
# Model learnKNN with GridSearchCV...
# Fitting 5 folds for each of 18 candidates, totalling 90 fits
# [CV] END ....................metric=euclidean, n_neighbors=3; total time=   1.2s
# [CV] END ....................metric=euclidean, n_neighbors=3; total time=   1.2s
# [CV] END ....................metric=euclidean, n_neighbors=5; total time=   1.2s
# [CV] END ....................metric=euclidean, n_neighbors=7; total time=   1.2s
# [CV] END ....................metric=euclidean, n_neighbors=5; total time=   1.3s
# [CV] END ....................metric=euclidean, n_neighbors=7; total time=   1.3s
# [CV] END ....................metric=euclidean, n_neighbors=9; total time=   1.3s
# [CV] END ....................metric=euclidean, n_neighbors=3; total time=   1.3s
# [CV] END ....................metric=euclidean, n_neighbors=9; total time=   1.4s
# [CV] END ....................metric=euclidean, n_neighbors=7; total time=   1.5s
# [CV] END ....................metric=euclidean, n_neighbors=5; total time=   1.6s
# [CV] END ....................metric=euclidean, n_neighbors=3; total time=   1.5s
# [CV] END ....................metric=euclidean, n_neighbors=7; total time=   1.5s
# [CV] END ....................metric=euclidean, n_neighbors=7; total time=   1.6s
# [CV] END ....................metric=euclidean, n_neighbors=9; total time=   1.4s
# [CV] END ....................metric=euclidean, n_neighbors=9; total time=   1.5s
# [CV] END ....................metric=euclidean, n_neighbors=3; total time=   1.7s
# [CV] END ....................metric=euclidean, n_neighbors=5; total time=   1.8s
# [CV] END ....................metric=euclidean, n_neighbors=9; total time=   1.9s
# [CV] END ....................metric=euclidean, n_neighbors=5; total time=   2.0s
# [CV] END ...................metric=euclidean, n_neighbors=11; total time=   1.2s
# [CV] END ...................metric=euclidean, n_neighbors=11; total time=   1.2s
# [CV] END ...................metric=euclidean, n_neighbors=11; total time=   1.2s
# [CV] END ...................metric=euclidean, n_neighbors=11; total time=   1.3s
# [CV] END ...................metric=euclidean, n_neighbors=13; total time=   1.3s
# [CV] END ...................metric=euclidean, n_neighbors=13; total time=   1.3s
# [CV] END ...................metric=euclidean, n_neighbors=13; total time=   1.3s
# [CV] END ...................metric=euclidean, n_neighbors=11; total time=   1.8s
# [CV] END ...................metric=euclidean, n_neighbors=13; total time=   1.5s
# [CV] END ...................metric=euclidean, n_neighbors=13; total time=   1.7s
# [CV] END ...................metric=euclidean, n_neighbors=15; total time=   1.5s
# [CV] END ...................metric=euclidean, n_neighbors=17; total time=   1.4s
# [CV] END ...................metric=euclidean, n_neighbors=15; total time=   1.6s
# [CV] END ...................metric=euclidean, n_neighbors=17; total time=   1.5s
# [CV] END ...................metric=euclidean, n_neighbors=17; total time=   1.4s
# [CV] END ...................metric=euclidean, n_neighbors=17; total time=   1.4s
# [CV] END ...................metric=euclidean, n_neighbors=15; total time=   1.7s
# [CV] END ...................metric=euclidean, n_neighbors=15; total time=   1.7s
# [CV] END ...................metric=euclidean, n_neighbors=15; total time=   2.0s
# [CV] END ...................metric=euclidean, n_neighbors=19; total time=   1.2s
# [CV] END ...................metric=euclidean, n_neighbors=19; total time=   1.2s
# [CV] END ...................metric=euclidean, n_neighbors=17; total time=   1.8s
# [CV] END ....................metric=manhattan, n_neighbors=3; total time=   1.1s
# [CV] END ....................metric=manhattan, n_neighbors=3; total time=   1.1s
# [CV] END ...................metric=euclidean, n_neighbors=19; total time=   1.5s
# [CV] END ...................metric=euclidean, n_neighbors=19; total time=   1.8s
# [CV] END ...................metric=euclidean, n_neighbors=19; total time=   1.8s
# [CV] END ....................metric=manhattan, n_neighbors=5; total time=   1.2s
# [CV] END ....................metric=manhattan, n_neighbors=5; total time=   1.1s
# [CV] END ....................metric=manhattan, n_neighbors=5; total time=   1.2s
# [CV] END ....................metric=manhattan, n_neighbors=3; total time=   1.2s
# [CV] END ....................metric=manhattan, n_neighbors=5; total time=   1.2s
# [CV] END ....................metric=manhattan, n_neighbors=5; total time=   1.2s
# [CV] END ....................metric=manhattan, n_neighbors=3; total time=   1.5s
# [CV] END ....................metric=manhattan, n_neighbors=7; total time=   1.2s
# [CV] END ....................metric=manhattan, n_neighbors=7; total time=   1.3s
# [CV] END ....................metric=manhattan, n_neighbors=7; total time=   1.3s
# [CV] END ....................metric=manhattan, n_neighbors=3; total time=   1.8s
# [CV] END ....................metric=manhattan, n_neighbors=9; total time=   1.2s
# [CV] END ....................metric=manhattan, n_neighbors=7; total time=   1.2s
# [CV] END ....................metric=manhattan, n_neighbors=7; total time=   1.6s
# [CV] END ....................metric=manhattan, n_neighbors=9; total time=   1.5s
# [CV] END ....................metric=manhattan, n_neighbors=9; total time=   1.1s
# [CV] END ...................metric=manhattan, n_neighbors=11; total time=   1.2s
# [CV] END ....................metric=manhattan, n_neighbors=9; total time=   1.6s
# [CV] END ...................metric=manhattan, n_neighbors=11; total time=   1.3s
# [CV] END ....................metric=manhattan, n_neighbors=9; total time=   1.8s
# [CV] END ...................metric=manhattan, n_neighbors=11; total time=   1.3s
# [CV] END ...................metric=manhattan, n_neighbors=13; total time=   1.2s
# [CV] END ...................metric=manhattan, n_neighbors=13; total time=   1.3s
# [CV] END ...................metric=manhattan, n_neighbors=11; total time=   1.5s
# [CV] END ...................metric=manhattan, n_neighbors=13; total time=   1.2s
# [CV] END ...................metric=manhattan, n_neighbors=15; total time=   1.3s
# [CV] END ...................metric=manhattan, n_neighbors=11; total time=   1.8s
# [CV] END ...................metric=manhattan, n_neighbors=13; total time=   1.5s
# [CV] END ...................metric=manhattan, n_neighbors=13; total time=   1.8s
# [CV] END ...................metric=manhattan, n_neighbors=15; total time=   1.2s
# [CV] END ...................metric=manhattan, n_neighbors=15; total time=   1.7s
# [CV] END ...................metric=manhattan, n_neighbors=15; total time=   1.6s
# [CV] END ...................metric=manhattan, n_neighbors=15; total time=   1.4s
# [CV] END ...................metric=manhattan, n_neighbors=17; total time=   1.2s
# [CV] END ...................metric=manhattan, n_neighbors=17; total time=   1.1s
# [CV] END ...................metric=manhattan, n_neighbors=17; total time=   1.3s
# [CV] END ...................metric=manhattan, n_neighbors=17; total time=   1.0s
# [CV] END ...................metric=manhattan, n_neighbors=19; total time=   1.0s
# [CV] END ...................metric=manhattan, n_neighbors=17; total time=   1.1s
# [CV] END ...................metric=manhattan, n_neighbors=19; total time=   1.0s
# [CV] END ...................metric=manhattan, n_neighbors=19; total time=   1.0s
# [CV] END ...................metric=manhattan, n_neighbors=19; total time=   1.0s
# [CV] END ...................metric=manhattan, n_neighbors=19; total time=   1.1s
# Best params for KNN: {'metric': 'euclidean', 'n_neighbors': 19}
# Best F1-score: 0.7687

# 🔹 KNN result:
#               precision    recall  f1-score   support

#            0      0.772     0.750     0.761     19472
#            1      0.757     0.778     0.768     19495

#     accuracy                          0.764     38967
#    macro avg      0.764     0.764     0.764     38967
# weighted avg      0.764     0.764     0.764     38967

# Accuracy: 0.7641, Precision: 0.7570, Recall: 0.7784, F1: 0.7675
# ROC-AUC: 0.8350
# ROC-chart save: artifacts\plots\roc_KNN.png
# Best model KNN was saved in: artifacts\models\KNN_best.joblib
# Training models:  25%|████████████████████████████████████████▌                                                                                                                         | 1/4 [00:10<00:31, 10.65s/it]
# Model learnDecisionTree with GridSearchCV...
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# [CV] END ........................criterion=gini, max_depth=3; total time=   0.2s
# [CV] END ........................criterion=gini, max_depth=3; total time=   0.2s
# [CV] END ........................criterion=gini, max_depth=3; total time=   0.3s
# [CV] END ........................criterion=gini, max_depth=3; total time=   0.3s
# [CV] END ........................criterion=gini, max_depth=3; total time=   0.3s
# [CV] END ........................criterion=gini, max_depth=5; total time=   0.4s
# [CV] END ........................criterion=gini, max_depth=5; total time=   0.4s
# [CV] END ........................criterion=gini, max_depth=5; total time=   0.4s
# [CV] END ........................criterion=gini, max_depth=5; total time=   0.4s
# [CV] END ........................criterion=gini, max_depth=5; total time=   0.5s
# [CV] END ........................criterion=gini, max_depth=7; total time=   0.6s
# [CV] END ........................criterion=gini, max_depth=7; total time=   0.6s
# [CV] END ........................criterion=gini, max_depth=7; total time=   0.6s
# [CV] END ........................criterion=gini, max_depth=7; total time=   0.7s
# [CV] END ........................criterion=gini, max_depth=7; total time=   0.6s
# [CV] END .....................criterion=entropy, max_depth=3; total time=   0.3s
# [CV] END .....................criterion=entropy, max_depth=3; total time=   0.3s
# [CV] END .....................criterion=entropy, max_depth=3; total time=   0.3s
# [CV] END .......................criterion=gini, max_depth=10; total time=   0.8s
# [CV] END .....................criterion=entropy, max_depth=3; total time=   0.3s
# [CV] END .....................criterion=entropy, max_depth=3; total time=   0.3s
# [CV] END .......................criterion=gini, max_depth=10; total time=   0.9s
# [CV] END .......................criterion=gini, max_depth=10; total time=   0.8s
# [CV] END .......................criterion=gini, max_depth=10; total time=   0.9s
# [CV] END .......................criterion=gini, max_depth=10; total time=   0.9s
# [CV] END .....................criterion=entropy, max_depth=5; total time=   0.6s
# [CV] END .....................criterion=entropy, max_depth=5; total time=   0.6s
# [CV] END .....................criterion=entropy, max_depth=5; total time=   0.6s
# [CV] END .....................criterion=entropy, max_depth=5; total time=   0.7s
# [CV] END .....................criterion=entropy, max_depth=5; total time=   0.7s
# [CV] END .....................criterion=entropy, max_depth=7; total time=   0.8s
# [CV] END .....................criterion=entropy, max_depth=7; total time=   0.8s
# [CV] END .....................criterion=entropy, max_depth=7; total time=   0.8s
# [CV] END .....................criterion=entropy, max_depth=7; total time=   0.8s
# [CV] END .....................criterion=entropy, max_depth=7; total time=   0.8s
# [CV] END ....................criterion=entropy, max_depth=10; total time=   1.0s
# [CV] END ....................criterion=entropy, max_depth=10; total time=   1.0s
# [CV] END ....................criterion=entropy, max_depth=10; total time=   1.1s
# [CV] END ....................criterion=entropy, max_depth=10; total time=   1.1s
# [CV] END ....................criterion=entropy, max_depth=10; total time=   1.1s
# [CV] END .....................criterion=gini, max_depth=None; total time=   1.8s
# [CV] END .....................criterion=gini, max_depth=None; total time=   1.9s
# [CV] END .....................criterion=gini, max_depth=None; total time=   2.0s
# [CV] END .....................criterion=gini, max_depth=None; total time=   2.1s
# [CV] END .....................criterion=gini, max_depth=None; total time=   2.2s
# [CV] END ..................criterion=entropy, max_depth=None; total time=   2.2s
# [CV] END ..................criterion=entropy, max_depth=None; total time=   2.4s
# [CV] END ..................criterion=entropy, max_depth=None; total time=   2.5s
# [CV] END ..................criterion=entropy, max_depth=None; total time=   2.6s
# [CV] END ..................criterion=entropy, max_depth=None; total time=   2.8s
# Best params for DecisionTree: {'criterion': 'gini', 'max_depth': 5}
# Best F1-score: 0.7720

# 🔹 DecisionTree result:
#               precision    recall  f1-score   support

#            0      0.769     0.784     0.777     19472
#            1      0.780     0.765     0.772     19495

#     accuracy                          0.774     38967
#    macro avg      0.775     0.774     0.774     38967
# weighted avg      0.775     0.774     0.774     38967

# Accuracy: 0.7744, Precision: 0.7802, Recall: 0.7646, F1: 0.7723
# ROC-AUC: 0.8604
# ROC-chart save: artifacts\plots\roc_DecisionTree.png
# Best model DecisionTree was saved in: artifacts\models\DecisionTree_best.joblib
# Training models:  50%|█████████████████████████████████████████████████████████████████████████████████                                                                                 | 2/4 [00:16<00:15,  7.55s/it]
# Model learnRandomForest with GridSearchCV...
# Fitting 5 folds for each of 27 candidates, totalling 135 fits
# [CV] END ...max_depth=5, min_samples_leaf=1, n_estimators=50; total time=   8.5s
# [CV] END ...max_depth=5, min_samples_leaf=1, n_estimators=50; total time=   9.0s
# [CV] END ...max_depth=5, min_samples_leaf=1, n_estimators=50; total time=   9.0s
# [CV] END ...max_depth=5, min_samples_leaf=1, n_estimators=50; total time=   9.1s
# [CV] END ...max_depth=5, min_samples_leaf=3, n_estimators=50; total time=   9.1s
# [CV] END ...max_depth=5, min_samples_leaf=3, n_estimators=50; total time=   9.1s
# [CV] END ...max_depth=5, min_samples_leaf=3, n_estimators=50; total time=   9.2s
# [CV] END ...max_depth=5, min_samples_leaf=3, n_estimators=50; total time=   9.6s
# [CV] END ...max_depth=5, min_samples_leaf=1, n_estimators=50; total time=   9.7s
# [CV] END ...max_depth=5, min_samples_leaf=3, n_estimators=50; total time=   9.7s
# [CV] END ..max_depth=5, min_samples_leaf=1, n_estimators=100; total time=  18.0s
# [CV] END ..max_depth=5, min_samples_leaf=1, n_estimators=100; total time=  18.1s
# [CV] END ..max_depth=5, min_samples_leaf=1, n_estimators=100; total time=  18.2s
# [CV] END ..max_depth=5, min_samples_leaf=1, n_estimators=100; total time=  19.0s
# [CV] END ..max_depth=5, min_samples_leaf=1, n_estimators=100; total time=  20.5s
# [CV] END ...max_depth=5, min_samples_leaf=5, n_estimators=50; total time=   9.7s
# [CV] END ..max_depth=5, min_samples_leaf=3, n_estimators=100; total time=  19.6s
# [CV] END ...max_depth=5, min_samples_leaf=5, n_estimators=50; total time=  10.2s
# [CV] END ..max_depth=5, min_samples_leaf=3, n_estimators=100; total time=  19.2s
# [CV] END ...max_depth=5, min_samples_leaf=5, n_estimators=50; total time=  10.3s
# [CV] END ..max_depth=5, min_samples_leaf=3, n_estimators=100; total time=  19.7s
# [CV] END ..max_depth=5, min_samples_leaf=3, n_estimators=100; total time=  20.3s
# [CV] END ..max_depth=5, min_samples_leaf=3, n_estimators=100; total time=  20.6s
# [CV] END ...max_depth=5, min_samples_leaf=5, n_estimators=50; total time=  10.6s
# [CV] END ...max_depth=5, min_samples_leaf=5, n_estimators=50; total time=  10.5s
# [CV] END ..max_depth=5, min_samples_leaf=1, n_estimators=200; total time=  38.3s
# [CV] END ..max_depth=5, min_samples_leaf=1, n_estimators=200; total time=  38.5s
# [CV] END ..max_depth=5, min_samples_leaf=1, n_estimators=200; total time=  39.1s
# [CV] END ..max_depth=5, min_samples_leaf=1, n_estimators=200; total time=  39.3s
# [CV] END ..max_depth=5, min_samples_leaf=1, n_estimators=200; total time=  40.3s
# [CV] END ..max_depth=5, min_samples_leaf=5, n_estimators=100; total time=  19.3s
# [CV] END ..max_depth=5, min_samples_leaf=5, n_estimators=100; total time=  20.0s
# [CV] END ..max_depth=5, min_samples_leaf=5, n_estimators=100; total time=  19.9s
# [CV] END ..max_depth=5, min_samples_leaf=5, n_estimators=100; total time=  20.2s
# [CV] END ..max_depth=5, min_samples_leaf=3, n_estimators=200; total time=  39.8s
# [CV] END ..max_depth=5, min_samples_leaf=3, n_estimators=200; total time=  39.5s
# [CV] END ..max_depth=5, min_samples_leaf=5, n_estimators=100; total time=  21.0s
# [CV] END ..max_depth=5, min_samples_leaf=3, n_estimators=200; total time=  40.9s
# [CV] END ..max_depth=5, min_samples_leaf=3, n_estimators=200; total time=  41.0s
# [CV] END ..max_depth=5, min_samples_leaf=3, n_estimators=200; total time=  41.8s
# [CV] END ..max_depth=10, min_samples_leaf=1, n_estimators=50; total time=  16.7s
# [CV] END ..max_depth=10, min_samples_leaf=1, n_estimators=50; total time=  16.7s
# [CV] END ..max_depth=10, min_samples_leaf=1, n_estimators=50; total time=  17.0s
# [CV] END ..max_depth=10, min_samples_leaf=1, n_estimators=50; total time=  17.0s
# [CV] END ..max_depth=10, min_samples_leaf=1, n_estimators=50; total time=  16.7s
# [CV] END ..max_depth=5, min_samples_leaf=5, n_estimators=200; total time=  40.1s
# [CV] END ..max_depth=5, min_samples_leaf=5, n_estimators=200; total time=  39.8s
# [CV] END ..max_depth=5, min_samples_leaf=5, n_estimators=200; total time=  41.8s
# [CV] END ..max_depth=5, min_samples_leaf=5, n_estimators=200; total time=  40.3s
# [CV] END ..max_depth=5, min_samples_leaf=5, n_estimators=200; total time=  42.1s
# [CV] END ..max_depth=10, min_samples_leaf=3, n_estimators=50; total time=  17.8s
# [CV] END ..max_depth=10, min_samples_leaf=3, n_estimators=50; total time=  18.4s
# [CV] END ..max_depth=10, min_samples_leaf=3, n_estimators=50; total time=  17.8s
# [CV] END ..max_depth=10, min_samples_leaf=3, n_estimators=50; total time=  17.0s
# [CV] END ..max_depth=10, min_samples_leaf=3, n_estimators=50; total time=  18.2s
# [CV] END .max_depth=10, min_samples_leaf=1, n_estimators=100; total time=  36.5s
# [CV] END .max_depth=10, min_samples_leaf=1, n_estimators=100; total time=  35.8s
# [CV] END .max_depth=10, min_samples_leaf=1, n_estimators=100; total time=  36.2s
# [CV] END .max_depth=10, min_samples_leaf=1, n_estimators=100; total time=  35.7s
# [CV] END .max_depth=10, min_samples_leaf=1, n_estimators=100; total time=  38.0s
# [CV] END ..max_depth=10, min_samples_leaf=5, n_estimators=50; total time=  17.5s
# [CV] END ..max_depth=10, min_samples_leaf=5, n_estimators=50; total time=  17.3s
# [CV] END ..max_depth=10, min_samples_leaf=5, n_estimators=50; total time=  18.1s
# [CV] END ..max_depth=10, min_samples_leaf=5, n_estimators=50; total time=  18.9s
# [CV] END .max_depth=10, min_samples_leaf=3, n_estimators=100; total time=  35.7s
# [CV] END ..max_depth=10, min_samples_leaf=5, n_estimators=50; total time=  18.9s
# [CV] END .max_depth=10, min_samples_leaf=3, n_estimators=100; total time=  35.4s
# [CV] END .max_depth=10, min_samples_leaf=3, n_estimators=100; total time=  38.9s
# [CV] END .max_depth=10, min_samples_leaf=3, n_estimators=100; total time=  36.4s
# [CV] END .max_depth=10, min_samples_leaf=3, n_estimators=100; total time=  37.2s
# [CV] END .max_depth=10, min_samples_leaf=1, n_estimators=200; total time= 1.1min
# [CV] END .max_depth=10, min_samples_leaf=1, n_estimators=200; total time= 1.2min
# [CV] END .max_depth=10, min_samples_leaf=1, n_estimators=200; total time= 1.2min
# [CV] END .max_depth=10, min_samples_leaf=1, n_estimators=200; total time= 1.2min
# [CV] END .max_depth=10, min_samples_leaf=1, n_estimators=200; total time= 1.2min
# [CV] END .max_depth=10, min_samples_leaf=5, n_estimators=100; total time=  32.5s
# [CV] END .max_depth=10, min_samples_leaf=5, n_estimators=100; total time=  34.4s
# [CV] END .max_depth=10, min_samples_leaf=5, n_estimators=100; total time=  34.4s
# [CV] END .max_depth=10, min_samples_leaf=5, n_estimators=100; total time=  34.0s
# [CV] END .max_depth=10, min_samples_leaf=5, n_estimators=100; total time=  34.2s
# [CV] END .max_depth=10, min_samples_leaf=3, n_estimators=200; total time= 1.1min
# [CV] END .max_depth=10, min_samples_leaf=3, n_estimators=200; total time= 1.1min
# [CV] END .max_depth=10, min_samples_leaf=3, n_estimators=200; total time= 1.1min
# [CV] END .max_depth=10, min_samples_leaf=3, n_estimators=200; total time= 1.2min
# [CV] END .max_depth=10, min_samples_leaf=3, n_estimators=200; total time= 1.2min
# [CV] END max_depth=None, min_samples_leaf=1, n_estimators=50; total time=  35.2s
# [CV] END max_depth=None, min_samples_leaf=1, n_estimators=50; total time=  34.7s
# [CV] END max_depth=None, min_samples_leaf=1, n_estimators=50; total time=  35.1s
# [CV] END max_depth=None, min_samples_leaf=1, n_estimators=50; total time=  35.0s
# [CV] END max_depth=None, min_samples_leaf=1, n_estimators=50; total time=  34.9s
# [CV] END .max_depth=10, min_samples_leaf=5, n_estimators=200; total time= 1.1min
# [CV] END .max_depth=10, min_samples_leaf=5, n_estimators=200; total time= 1.1min
# [CV] END .max_depth=10, min_samples_leaf=5, n_estimators=200; total time= 1.1min
# [CV] END .max_depth=10, min_samples_leaf=5, n_estimators=200; total time= 1.1min
# [CV] END .max_depth=10, min_samples_leaf=5, n_estimators=200; total time= 1.1min
# [CV] END max_depth=None, min_samples_leaf=3, n_estimators=50; total time=  32.5s
# [CV] END max_depth=None, min_samples_leaf=3, n_estimators=50; total time=  33.0s
# [CV] END max_depth=None, min_samples_leaf=3, n_estimators=50; total time=  34.1s
# [CV] END max_depth=None, min_samples_leaf=3, n_estimators=50; total time=  32.5s
# [CV] END max_depth=None, min_samples_leaf=3, n_estimators=50; total time=  34.8s
# [CV] END max_depth=None, min_samples_leaf=1, n_estimators=100; total time= 1.1min
# [CV] END max_depth=None, min_samples_leaf=1, n_estimators=100; total time= 1.1min
# [CV] END max_depth=None, min_samples_leaf=1, n_estimators=100; total time= 1.2min
# [CV] END max_depth=None, min_samples_leaf=1, n_estimators=100; total time= 1.2min
# [CV] END max_depth=None, min_samples_leaf=1, n_estimators=100; total time= 1.2min
# [CV] END max_depth=None, min_samples_leaf=3, n_estimators=100; total time= 1.0min
# [CV] END max_depth=None, min_samples_leaf=3, n_estimators=100; total time= 1.0min
# [CV] END max_depth=None, min_samples_leaf=5, n_estimators=50; total time=  28.9s
# [CV] END max_depth=None, min_samples_leaf=5, n_estimators=50; total time=  30.2s
# [CV] END max_depth=None, min_samples_leaf=3, n_estimators=100; total time= 1.1min
# [CV] END max_depth=None, min_samples_leaf=5, n_estimators=50; total time=  31.7s
# [CV] END max_depth=None, min_samples_leaf=5, n_estimators=50; total time=  30.8s
# [CV] END max_depth=None, min_samples_leaf=3, n_estimators=100; total time= 1.1min
# [CV] END max_depth=None, min_samples_leaf=3, n_estimators=100; total time= 1.1min
# [CV] END max_depth=None, min_samples_leaf=5, n_estimators=50; total time=  30.3s
# [CV] END max_depth=None, min_samples_leaf=1, n_estimators=200; total time= 2.2min
# [CV] END max_depth=None, min_samples_leaf=1, n_estimators=200; total time= 2.3min
# [CV] END max_depth=None, min_samples_leaf=1, n_estimators=200; total time= 2.3min
# [CV] END max_depth=None, min_samples_leaf=1, n_estimators=200; total time= 2.3min
# [CV] END max_depth=None, min_samples_leaf=1, n_estimators=200; total time= 2.4min
# [CV] END max_depth=None, min_samples_leaf=5, n_estimators=100; total time=  56.9s
# [CV] END max_depth=None, min_samples_leaf=5, n_estimators=100; total time=  57.0s
# [CV] END max_depth=None, min_samples_leaf=5, n_estimators=100; total time=  59.5s
# [CV] END max_depth=None, min_samples_leaf=5, n_estimators=100; total time=  59.8s
# [CV] END max_depth=None, min_samples_leaf=5, n_estimators=100; total time=  58.6s
# [CV] END max_depth=None, min_samples_leaf=3, n_estimators=200; total time= 1.9min
# [CV] END max_depth=None, min_samples_leaf=3, n_estimators=200; total time= 2.0min
# [CV] END max_depth=None, min_samples_leaf=3, n_estimators=200; total time= 2.0min
# [CV] END max_depth=None, min_samples_leaf=3, n_estimators=200; total time= 2.0min
# [CV] END max_depth=None, min_samples_leaf=3, n_estimators=200; total time= 2.0min
# [CV] END max_depth=None, min_samples_leaf=5, n_estimators=200; total time= 1.5min
# [CV] END max_depth=None, min_samples_leaf=5, n_estimators=200; total time= 1.5min
# [CV] END max_depth=None, min_samples_leaf=5, n_estimators=200; total time= 1.5min
# [CV] END max_depth=None, min_samples_leaf=5, n_estimators=200; total time= 1.5min
# [CV] END max_depth=None, min_samples_leaf=5, n_estimators=200; total time= 1.5min
# Best params for RandomForest: {'max_depth': 10, 'min_samples_leaf': 5, 'n_estimators': 100}
# Best F1-score: 0.7706

# 🔹 RandomForest result:
#               precision    recall  f1-score   support

#            0      0.767     0.787     0.777     19472
#            1      0.782     0.762     0.771     19495

#     accuracy                          0.774     38967
#    macro avg      0.774     0.774     0.774     38967
# weighted avg      0.774     0.774     0.774     38967

# Accuracy: 0.7743, Precision: 0.7817, Recall: 0.7616, F1: 0.7715
# ROC-AUC: 0.8612
# ROC-chart save: artifacts\plots\roc_RandomForest.png
# Best model RandomForest was saved in: artifacts\models\RandomForest_best.joblib
# Training models:  75%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                        | 3/4 [06:09<02:45, 165.61s/it]
# Training SVM without GridSearch...

# 🔹 SVM result:
#               precision    recall  f1-score   support

#            0      0.707     0.858     0.775     19472
#            1      0.819     0.645     0.722     19495

#     accuracy                          0.751     38967
#    macro avg      0.763     0.751     0.748     38967
# weighted avg      0.763     0.751     0.748     38967

# Accuracy: 0.7513, Precision: 0.8194, Recall: 0.6450, F1: 0.7218
# ROC-AUC: 0.8435
# ROC-chart save: artifacts\plots\roc_SVM.png
# Best model SVM was saved in: artifacts\models\SVM_best.joblib
# Training models: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [06:10<00:00, 92.55s/it]