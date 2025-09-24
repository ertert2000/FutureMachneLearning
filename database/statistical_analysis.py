import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine 
from sqlalchemy.orm import sessionmaker
from models.Models import *
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, chi2_contingency, pointbiserialr

engine = create_engine("sqlite:///crypto.db")
Session = sessionmaker(bind=engine)
session = Session()

def outliers_indices(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return series[(series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)].index

trading_pairs = {
    1 : "BTC/USDT",
    2 : "ETH/USDT",
    3 : "BNB/USDT",
    4 : "XRP/USDT",
    5 : "ADA/USDT",
    6 : "SOL/USDT",
    7 : "OP/USDT",
    8 : "TRX/USDT", 
    9 : "DOT/USDT",
    10 : "MATIC/USDT"
}

for i in range(1, 11):
    query = session.query(Candle).join(Symbol).filter(
        Symbol.name == trading_pairs[i]
    )

    numeric = ["open", "high", "low", "close", "volume"]

    df = pd.DataFrame([{
        "date": x.date,
        "open": x.open,
        "high": x.high,
        "low": x.low,
        "close": x.close,
        "volume": x.volume
    } for x in query])

    # Первичный анализ
    print(f'{trading_pairs[i]} \n{df[numeric].describe()}')

    # удаление выбросов
    out = set()
    for col in numeric:
        out |= set(outliers_indices(df[col]))
    print(f"Будет удалено выбросов: {len(out)}\n\n")
    df_clean = df.drop(out)

    # кореляционный анализ
    sns.heatmap(df_clean[numeric].corr(method="spearman"), annot=True, cmap="coolwarm")
    plt.title("spearman")
    plt.show()

    #
    r = pearsonr(df_clean['close'], df_clean['volume'])
    print("Pearson:", r)

    r = spearmanr(df_clean['close'], df_clean['volume'])
    print("Spearman:", r)

    df_clean['high'] = (df_clean['volume'] > df_clean['volume'].median()).astype(int)
    r_pb = pointbiserialr(df_clean['high'], df_clean['close'])
    print(f"Point-biserial (high_vol vs close): r={r_pb.correlation:.3f}, p={r_pb.pvalue:.3e}")