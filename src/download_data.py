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


#mostrar datos
print("preview (head):")
print(data.head())
print("\nShape (rows, cols):", data.shape)
print("Type:", type(data))
print("Index:", data.index)
print("Columns:", list(data.columns))
print("\nClose + DailyReturn (primeras 10 filas):")
print(data[["Date", "Close", "DailyReturn"]].head(10))
print("\nDailyReturn stats:")
print(data["DailyReturn"].describe())

#guardar datos
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

filename = out_dir / f"{ticker}_{period}_clean.csv"
data.to_csv(filename, index=False)

print(f"\nData saved to: {filename}")