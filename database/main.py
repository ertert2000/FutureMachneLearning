import ccxt
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.Models import *

engine = create_engine("sqlite:///crypto.db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

exchange = ccxt.binance()

top10 = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
    "SOL/USDT", "OP/USDT", "TRX/USDT", "DOT/USDT", "MATIC/USDT"
]

timeframe = "1h"
limit = 1000
num_batches = 20 

writer = pd.ExcelWriter("data.xlsx", engine="openpyxl")

for symbol_name in top10:
    print(f"Load {symbol_name} ...")

    try:
        db_symbol = session.query(Symbol).filter_by(name=symbol_name).first()
        if not db_symbol:
            db_symbol = Symbol(name=symbol_name)
            session.add(db_symbol)
            session.commit()

        all_data = []
        since = exchange.parse8601("2023-01-01T00:00:00Z")

        for i in range(num_batches):
            ohlcv = exchange.fetch_ohlcv(symbol_name, timeframe=timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_data += ohlcv
            since = ohlcv[-1][0] + 1
    except Exception as e:
        print(e)

    try:
        df = pd.DataFrame(all_data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

        df.to_excel(writer, sheet_name=symbol_name.replace("/", "_"), index=False)


        candles = [
            Candle(
                symbol_id=db_symbol.id,
                timeframe=timeframe,
                date=row.Date,
                open=row.Open,
                high=row.High,
                low=row.Low,
                close=row.Close,
                volume=row.Volume
            )
            for row in df.itertuples()
        ]

    except Exception as e:
        print(e)
    try:
        session.bulk_save_objects(candles)
    except Exception as e:
        print(e)

    session.commit()

writer.close()
session.close()

print("complete")
