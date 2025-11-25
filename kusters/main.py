import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

DB_PATH = "sqlite:///crypto.db"
ARTIFACTS_DIR = "artifacts"
RANDOM_STATE = 42
AGGLO_SAMPLE_SIZE = 1000
DBSCAN_SAMPLE_SIZE = 10000
KMEANS_N_CLUSTERS = 5
TEST_SIZE = 0.2

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
sns.set(style="whitegrid")

from models.Models import Candle

def load_crypto_data():
    engine = create_engine(DB_PATH)
    Session = sessionmaker(bind=engine)
    session = Session()
    query = session.query(Candle).all()
    data = pd.DataFrame([
        {"date": x.date, "open": x.open, "high": x.high, "low": x.low, "close": x.close, "volume": x.volume}
        for x in query
    ])
    session.close()
    print(f"Loaded {len(data)} rows.")
    return data

def plot_cluster_scatter(data, x_col, y_col, cluster_col, centroids=None, title=None):
    plt.figure(figsize=(10, 6))

    unique_clusters = sorted(data[cluster_col].dropna().unique())

    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster in enumerate(unique_clusters):
        subset = data[data[cluster_col] == cluster]
        plt.scatter(
            subset[x_col], subset[y_col],
            s=20, color=colors[i], label=f"Кластер {cluster}", alpha=0.7
        )

    if centroids is not None:
        for i, c in enumerate(centroids):
            plt.scatter(c[0], c[1], color="red", marker="*", s=200, label="Центроид" if i==0 else "")

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title if title else f"Кластеры {x_col} / {y_col}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(ARTIFACTS_DIR, title if title else f"Кластеры {x_col} / {y_col}"))
    plt.show()

data = load_crypto_data()

data = data.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
print(f"After dropna: {len(data)} rows.")

features = ["open", "high", "low", "close", "volume"]
X = data[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.joblib"))

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
data["pca1"] = X_pca[:, 0]
data["pca2"] = X_pca[:, 1]
joblib.dump(pca, os.path.join(ARTIFACTS_DIR, "pca.joblib"))

n_samples_agglo = min(AGGLO_SAMPLE_SIZE, X_scaled.shape[0])
np.random.seed(RANDOM_STATE)
sample_idx_agglo = np.random.choice(X_scaled.shape[0], n_samples_agglo, replace=False)
X_sample_agglo = X_scaled[sample_idx_agglo]

sample_idx = np.random.choice(X_scaled.shape[0], 1500, replace=False)
X_sample = X_scaled[sample_idx]

Z = linkage(X_sample, method='ward')

plt.figure(figsize=(14, 7))
plt.title("Дендрограмма иерархической кластеризации (Ward)")
plt.xlabel("Объекты выборки")
plt.ylabel("Евклидово расстояние")
dendrogram(Z, truncate_mode="level", p=6)
plt.grid(True)
plt.savefig(os.path.join(ARTIFACTS_DIR, "denda.png"))
plt.show()

agglo = AgglomerativeClustering(n_clusters=3)
clusters_agglo = agglo.fit_predict(X_sample_agglo)

data["cluster_agglo"] = np.nan
data.loc[sample_idx_agglo, "cluster_agglo"] = clusters_agglo

plt.figure(figsize=(8,6))
plt.title("AgglomerativeClustering (PCA projection) - sample")
sns.scatterplot(x=data.loc[sample_idx_agglo, "pca1"],
                y=data.loc[sample_idx_agglo, "pca2"],
                hue=clusters_agglo, palette="tab10", legend="full")
plt.savefig(os.path.join(ARTIFACTS_DIR, "agglo_pca_scatter.png"))
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x=clusters_agglo)
plt.title("Agglo cluster counts (sample)")
plt.xlabel("cluster")
plt.ylabel("count")
plt.savefig(os.path.join(ARTIFACTS_DIR, "agglo_counts.png"))
plt.show()

plot_cluster_scatter(
    data,
    x_col="open",
    y_col="close",
    cluster_col="cluster_agglo",
    centroids=None,
    title="Agglomerative — Open Close"
)

plot_cluster_scatter(
    data,
    x_col="open",
    y_col="volume",
    cluster_col="cluster_agglo",
    title="Agglomerative — Open Volume"
)

n_samples_dbscan = min(DBSCAN_SAMPLE_SIZE, X_scaled.shape[0])
sample_idx_dbscan = np.random.choice(X_scaled.shape[0], n_samples_dbscan, replace=False)
X_sample_dbscan = X_scaled[sample_idx_dbscan]

dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters_dbscan = dbscan.fit_predict(X_sample_dbscan)

data["cluster_dbscan"] = np.nan
data.loc[sample_idx_dbscan, "cluster_dbscan"] = clusters_dbscan

plt.figure(figsize=(8,6))
plt.title("DBSCAN (PCA projection) - sample")
sns.scatterplot(x=data.loc[sample_idx_dbscan, "pca1"],
                y=data.loc[sample_idx_dbscan, "pca2"],
                hue=clusters_dbscan, palette="tab10", legend="full")
plt.savefig(os.path.join(ARTIFACTS_DIR, "dbscan_pca_scatter.png"))
plt.show()

plt.figure(figsize=(6,4))
unique, counts = np.unique(clusters_dbscan, return_counts=True)
plt.bar(unique.astype(str), counts)
plt.title("DBSCAN cluster counts (sample)")
plt.xlabel("cluster label (-1 = noise)")
plt.ylabel("count")
plt.savefig(os.path.join(ARTIFACTS_DIR, "dbscan_counts.png"))
plt.show()

plot_cluster_scatter(
    data,
    x_col="open",
    y_col="close",
    cluster_col="cluster_dbscan",
    title="DBSCAN — Open Close"
)

kmeans = KMeans(n_clusters=KMEANS_N_CLUSTERS, random_state=RANDOM_STATE)
clusters_kmeans = kmeans.fit_predict(X_scaled)
data["cluster_kmeans"] = clusters_kmeans
joblib.dump(kmeans, os.path.join(ARTIFACTS_DIR, "kmeans.joblib"))

plt.figure(figsize=(8,6))
plt.title(f"KMeans (n={KMEANS_N_CLUSTERS}) - PCA projection (all)")
sns.scatterplot(x="pca1", y="pca2", hue="cluster_kmeans", data=data, palette="tab10", legend="full")
plt.savefig(os.path.join(ARTIFACTS_DIR, "kmeans_pca_scatter.png"))
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="cluster_kmeans", data=data)
plt.title("KMeans cluster counts (all)")
plt.xlabel("cluster")
plt.ylabel("count")
plt.savefig(os.path.join(ARTIFACTS_DIR, "kmeans_counts.png"))
plt.show()

centroids = None

plot_cluster_scatter(
    data.assign(centroid_x=centroids[:, 0], centroid_y=centroids[:, 1]),
    x_col="open",
    y_col="close",
    cluster_col="cluster_kmeans",
    centroids=centroids[:, [0, 3]],
    title="Кластеры KMeans — Open Close"
)

plot_cluster_scatter(
    data,
    x_col="open",
    y_col="volume",
    cluster_col="cluster_kmeans",
    centroids=centroids[:, [0, 4]],
    title="Кластеры KMeans — Open Volume"
)

plot_cluster_scatter(
    data,
    x_col="high",
    y_col="low",
    cluster_col="cluster_kmeans",
    centroids=centroids[:, [1, 2]],
    title="Кластеры KMeans — High Low"
)


pairplot_cols = ["open", "close", "volume", "cluster_kmeans"]
pairplot_sample = data[pairplot_cols].dropna().sample(frac=1.0 if len(data)<=2000 else 2000/len(data), random_state=RANDOM_STATE)
pairplot_sample["cluster_kmeans"] = pairplot_sample["cluster_kmeans"].astype(int).astype(str)
sns.pairplot(pairplot_sample, hue="cluster_kmeans", vars=["open", "close", "volume"], plot_kws={"alpha":0.6})
plt.suptitle("Pairplot (sample) — KMeans clusters", y=1.02)
plt.savefig(os.path.join(ARTIFACTS_DIR, "pairplot_kmeans.png"))
plt.show()

data.to_csv(os.path.join(ARTIFACTS_DIR, "data_with_clusters.csv"), index=False)
print(f"Saved data_with_clusters.csv ({len(data)} rows)")

X_reg = data[["open", "high", "low", "volume"]].copy()
y_reg = data["close"].copy()

reg_df = pd.concat([X_reg, y_reg], axis=1).dropna().reset_index(drop=True)
X_reg = reg_df[["open", "high", "low", "volume"]].values
y_reg = reg_df["close"].values

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE)

scaler_reg = StandardScaler()
X_train_s = scaler_reg.fit_transform(X_train)
X_test_s = scaler_reg.transform(X_test)
joblib.dump(scaler_reg, os.path.join(ARTIFACTS_DIR, "scaler_reg.joblib"))

def regression_report_and_plots(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n=== {model_name} ===")
    print(f"R^2:  {r2:.4f}")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")

    plt.figure(figsize=(7,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', label="y = x")
    plt.xlabel("Actual close")
    plt.ylabel("Predicted close")
    plt.title(f"{model_name} — Predicted vs Actual")
    plt.legend()
    plt.savefig(os.path.join(ARTIFACTS_DIR, f"{model_name}_pred_vs_actual.png"))
    plt.show()

    plt.figure(figsize=(7,5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Pred)")
    plt.title(f"{model_name} — Residuals vs Predicted")
    plt.savefig(os.path.join(ARTIFACTS_DIR, f"{model_name}_residuals_vs_pred.png"))
    plt.show()

    plt.figure(figsize=(7,5))
    sns.histplot(residuals, kde=True)
    plt.title(f"{model_name} — Residuals distribution")
    plt.xlabel("Residual")
    plt.savefig(os.path.join(ARTIFACTS_DIR, f"{model_name}_residuals_hist.png"))
    plt.show()

    if hasattr(model, "coef_"):
        coefs = model.coef_
        features = ["open", "high", "low", "volume"]
        coef_df = pd.DataFrame({"feature": features, "coef": coefs})
        coef_df = coef_df.sort_values("coef", key=lambda s: np.abs(s), ascending=False)
        plt.figure(figsize=(7,4))
        sns.barplot(x="coef", y="feature", data=coef_df)
        plt.title(f"{model_name} — Coefficients")
        plt.savefig(os.path.join(ARTIFACTS_DIR, f"{model_name}_coefficients.png"))
        plt.show()
        coef_df.to_csv(os.path.join(ARTIFACTS_DIR, f"{model_name}_coefficients.csv"), index=False)
    else:
        print(f"{model_name} has no coef_ attribute")

    res_df = pd.DataFrame({
        "actual": y_test,
        "predicted": y_pred,
        "residual": residuals
    })
    res_df.to_csv(os.path.join(ARTIFACTS_DIR, f"{model_name}_predictions.csv"), index=False)

    return {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae}

models = {
    "LinearRegression": LinearRegression(),
    "LassoCV": LassoCV(cv=5, random_state=RANDOM_STATE),
    "RidgeCV": RidgeCV(cv=5)
}

metrics_summary = {}

for name, model in models.items():
    metrics = regression_report_and_plots(model, X_train_s, y_train, X_test_s, y_test, name)
    metrics_summary[name] = metrics
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, f"{name}.joblib"))

metrics_df = pd.DataFrame(metrics_summary).T
metrics_df.to_csv(os.path.join(ARTIFACTS_DIR, "regression_metrics_summary.csv"))
print("\nRegression metrics summary:")
print(metrics_df)

full_reg_df = pd.concat([data[["date"] + features], data[["cluster_agglo", "cluster_dbscan", "cluster_kmeans"]]], axis=1)
full_reg_df = full_reg_df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)

X_full = full_reg_df[["open","high","low","volume"]].values
X_full_s = scaler_reg.transform(X_full)
lin_model = models["LinearRegression"]
full_preds = lin_model.predict(X_full_s)
full_reg_df["pred_close_linear"] = full_preds
full_reg_df.to_csv(os.path.join(ARTIFACTS_DIR, "full_data_with_preds.csv"), index=False)
print(f"Saved full_data_with_preds.csv ({len(full_reg_df)} rows)")

print("\nAll artifacts saved to:", ARTIFACTS_DIR)

# === LinearRegression ===
# R^2:  1.0000
# MSE:  2538.032057
# RMSE: 50.378885
# MAE:  9.594484

# === LassoCV ===
# R^2:  1.0000
# MSE:  12055.585264
# RMSE: 109.797929
# MAE:  25.272059

# === RidgeCV ===
# R^2:  1.0000
# MSE:  2513.535098
# RMSE: 50.135168
# MAE:  9.666551
