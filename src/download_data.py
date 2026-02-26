import yfinance as yf
import pandas as pd
from pathlib import Path

#configuracion 

ticker = "AAPL"
period = "6mo"

#descargar datos
data = yf.download(ticker, period=period, progress=False)

#caso multiindex

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

#pasar date a columna
data = data.reset_index()
data.columns.name = None

#retorno diario
data["DailyReturn"] = data["Close"].pct_change()

#porcentaje de retorno diario
data["DailyReturnPct"] = (data["DailyReturn"] * 100).round(3)

#volatilidad
data["Volatility20d"] = data["DailyReturn"].rolling(20).std()

#medidas moviles simples
data["SMA20"] = data["Close"].rolling(20).mean()
data["SMA50"] = data["Close"].rolling(50).mean()

#señal de cruce alcista
data["Signal"] = data["SMA20"] > data["SMA50"]

#cruce
data["CrossUp"] = ((data["SMA20"] > data["SMA50"]) & (data["SMA20"].shift(1) <= data["SMA50"].shift(1)))

#cruce bajista
data["CrossDown"] = ((data["SMA20"] < data["SMA50"]) & (data["SMA20"].shift(1) >= data["SMA50"].shift(1)))

#mostrar datos 
print("\n===== DATA SUMMARY =====")
print("Shape:", data.shape)
print("Columns:", list(data.columns))
print("\nPrimeras 5 filas:")
print(data.head())

print("\nDaily Return stats:")
print(data["DailyReturn"].describe())

print("\nCruces detectados:")
print("Golden Cross:", int(data["CrossUp"].sum()))
print("Death Cross:", int(data["CrossDown"].sum()))

#guardar datos
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

filename = out_dir / f"{ticker}_{period}_clean.csv"
data.to_csv(filename, index=False)

print(f"\nData saved to: {filename}")