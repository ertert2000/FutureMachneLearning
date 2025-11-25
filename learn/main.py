import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.Models import *
import joblib

DB_PATH = "sqlite:///crypto.db"
ARTIFACTS_DIR = "artifacts"
OUTPUT_FILE = "preprocessed_crypto_data.npz"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2


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



def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by="date").reset_index(drop=True)

    df["c1"] = df["close"].shift(2)
    df["c2"] = df["close"].shift(1)
    df["c3"] = df["close"]
    df["c4"] = df["close"].shift(-1)

    up_trend = (df["c3"] > df["c2"]) & (df["c2"] > df["c1"]) & (df["c4"] > df["c3"])
    down_trend = (df["c3"] < df["c2"]) & (df["c2"] < df["c1"]) & (df["c4"] < df["c3"])

    df["target"] = 0
    df.loc[up_trend, "target"] = 1
    df.loc[down_trend, "target"] = -1
    df = df.dropna().reset_index(drop=True)

    return df.drop(columns=["c1", "c2", "c3", "c4"])



def preprocess_and_save(df: pd.DataFrame):
    X = df[["open", "high", "low", "close", "volume"]]
    y = df["target"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    val_fraction = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_fraction, random_state=RANDOM_STATE, stratify=y_train_val
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    np.savez_compressed(
        OUTPUT_FILE,
        X_train=X_train_scaled,
        X_val=X_val_scaled,
        X_test=X_test_scaled,
        y_train=y_train.values,
        y_val=y_val.values,
        y_test=y_test.values
    )

    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.joblib"))

    print("\nComplete!")
    print(f"Scaler: {ARTIFACTS_DIR}/scaler.joblib")

if __name__ == "__main__":
    try:
        df = load_crypto_data()
        df = create_target(df)
        preprocess_and_save(df)
    except Exception as e:
        print(e)
