import pandas as pd
from sqlalchemy import create_engine 
from sqlalchemy.orm import sessionmaker
from models.Models import *
from sklearn import preprocessing

engine = create_engine("sqlite:///crypto.db")
Session = sessionmaker(bind=engine)
session = Session()

query = session.query(Candle).all()

data = pd.DataFrame([{
    "date": x.date,
    "open": x.open,
    "high": x.high,
    "low": x.low,
    "close": x.close,
    "volume": x.volume
} for x in query])

print(data.isna().sum())

for col in ["open", "high", "low", "close", "volume"]:
    if data[col].isna().sum() > 0:
        data[col] = data[col].fillna(data[col].median())

normalized_data = preprocessing.normalize(data[["open", "high", "low", "close", "volume"]])

print(normalized_data)

normalized_df = pd.DataFrame(normalized_data, columns=["open", "high", "low", "close", "volume"])
print(normalized_df.head())