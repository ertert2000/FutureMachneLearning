import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine 
from sqlalchemy.orm import sessionmaker
from models.Models import *

import plotly.graph_objects as go
import numpy as np



engine = create_engine("sqlite:///crypto.db")
Session = sessionmaker(bind=engine)
session = Session()

def price_to_volume(prices, volumes):
    plt.figure(figsize=(10, 7))
    plt.scatter(volumes, prices, color='blue', s=1)
    plt.xlabel('Trading volume, USTD')
    plt.ylabel('Close price, USDT')
    plt.title('Price vs Trade Volume')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_box_volume_matplotlib(symbol_name, timeframe='1h'):
    query = session.query(Candle).join(Symbol).filter(
        Symbol.name == symbol_name,
        Candle.timeframe == timeframe
    )
    df = pd.read_sql(query.statement, session.bind)

    plt.figure(figsize=(8,6))
    plt.boxplot(df['volume'], patch_artist=True, boxprops=dict(facecolor="lightgreen"), showfliers=True)
    plt.title(f'Boxplot: volumes differential {symbol_name}')
    plt.ylabel("Trading volumes, USDT")
    plt.grid(axis='y', linestyle="--", alpha=0.5)
    plt.show()

def plot_basic_histograms(symbol_name, timeframe='1h'):
    """Базовые гистограммы для анализа распределения"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
 
    # Получаем данные
    query = session.query(Candle).join(Symbol).filter(
        Symbol.name == symbol_name,
        Candle.timeframe == timeframe
    )
    df = pd.read_sql(query.statement, session.bind)
    
    # 1. Гистограмма цен закрытия
    axes[0, 0].hist(df['close'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title(f'Распределение цен закрытия {symbol_name}')
    axes[0, 0].set_xlabel('Цена')
    axes[0, 0].set_ylabel('Частота')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Гистограмма объемов
    axes[0, 1].hist(df['volume'], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title(f'Распределение объемов {symbol_name}')
    axes[0, 1].set_xlabel('Объем')
    axes[0, 1].set_ylabel('Частота')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Гистограмма дневной доходности
    returns = df['close'].pct_change().dropna() * 100
    axes[1, 0].hist(returns, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].set_title(f'Распределение дневной доходности {symbol_name}')
    axes[1, 0].set_xlabel('Доходность (%)')
    axes[1, 0].set_ylabel('Частота')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Гистограмма ценового диапазона
    price_range = (df['high'] - df['low']) / df['open'] * 100
    axes[1, 1].hist(price_range, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title(f'Распределение ценового диапазона {symbol_name}')
    axes[1, 1].set_xlabel('Диапазон (%)')
    axes[1, 1].set_ylabel('Частота')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    query = session.query(Candle).all()

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

    # TRADING VOLUME VS PRICE 
    # for i in range(1, 11):
    #     price = [x.close for x in query if x.symbol_id == i]
    #     volume = [x.volume for x in query if x.symbol_id == i]

    #     price_to_volume(price, volume)
    
    # BOXPLOT OF VOLUME
    # for i in range(1, 11):
    #     plot_box_volume_matplotlib(trading_pairs[i])

    # BOXPLOT OF VOLUME
    for i in range(1, 11):
        plot_basic_histograms(trading_pairs[i])