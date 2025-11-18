import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from models.Models import *

DB_PATH = "sqlite:///crypto.db"

def load_crypto_data():
    engine = create_engine(DB_PATH)
    Session = sessionmaker(bind=engine)
    session = Session()

    query = session.query(Candle).all()

    data = pd.DataFrame([
        {
            "date": x.date,
            "open": x.open,
            "high": x.high,
            "low": x.low,
            "close": x.close,
            "volume": x.volume
        }
        for x in query
    ])

    session.close()
    print(f"Loaded {len(data)}.")
    return data

data = load_crypto_data()

features = data[['open', 'high', 'low', 'close', 'volume']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

sample_idx_agglo = np.random.choice(X_scaled.shape[0], 1000, replace=False)
X_sample_agglo = X_scaled[sample_idx_agglo]

agglo = AgglomerativeClustering(n_clusters=3)
clusters_agglo = agglo.fit_predict(X_sample_agglo)

data['cluster_agglo'] = np.nan
data.loc[sample_idx_agglo, 'cluster_agglo'] = clusters_agglo

sample_idx_dbscan = np.random.choice(X_scaled.shape[0], 10000, replace=False)
X_sample_dbscan = X_scaled[sample_idx_dbscan]

dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters_dbscan = dbscan.fit_predict(X_sample_dbscan)

data['cluster_dbscan'] = np.nan
data.loc[sample_idx_dbscan, 'cluster_dbscan'] = clusters_dbscan

kmeans = KMeans(n_clusters=5, random_state=42)
data['cluster_kmeans'] = kmeans.fit_predict(X_scaled)

sns.pairplot(data[['open', 'close', 'volume', 'cluster_kmeans']], hue='cluster_kmeans')
plt.show()

X = data[['open', 'high', 'low', 'volume']]
y = data['close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("Linear Regression R^2:", lr_model.score(X_test, y_test))

lasso_model = LassoCV(cv=5)
lasso_model.fit(X_train, y_train)
print("LASSO R^2:", lasso_model.score(X_test, y_test))

ridge_model = RidgeCV(cv=5)
ridge_model.fit(X_train, y_train)
print("Ridge R^2:", ridge_model.score(X_test, y_test))