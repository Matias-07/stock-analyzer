import yfinance as yf
from pathlib import Path

ticker = "AAPL"
period = "6mo"

data = yf.download(ticker, period=period, progress=False)

#elimina el nivel de columnas
data.columns = data.columns.droplevel(1)

#convierte indice en columna
data = data.reset_index()
data.columns.name = None

print(data.head())
print(data.shape)
print(type(data))
print(data.index)
print(data.columns) 

Path("outputs").mkdir(exist_ok = True)
data.to_csv("outputs/AAPL_6mo_clean.csv", index=False)
print("saved output/AAPL_6mo_clean.csv")