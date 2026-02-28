import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ==============================
# Configuración
# ==============================
ticker = "AAPL"
period = "6mo"
transaction_cost = 0.001  # 0.1% por trade (entrada o salida)

out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

# ==============================
# Descargar datos
# ==============================
data = yf.download(ticker, period=period, progress=False)

if data.empty:
    raise ValueError(f"No se descargaron datos para ticker={ticker} period={period}")

# Caso MultiIndex (yfinance a veces devuelve columnas de 2 niveles)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# Pasar Date (índice) a columna
data = data.reset_index()
data.columns.name = None

# ==============================
# Features / indicadores
# ==============================
data["DailyReturn"] = data["Close"].pct_change()
data["DailyReturnPct"] = (data["DailyReturn"] * 100).round(3)

data["Volatility20d"] = data["DailyReturn"].rolling(20).std()

data["SMA20"] = data["Close"].rolling(20).mean()
data["SMA50"] = data["Close"].rolling(50).mean()

# Señal de estado (solo informativa)
data["Signal"] = data["SMA20"] > data["SMA50"]

# Cruces (eventos)
data["CrossUp"] = (data["SMA20"] > data["SMA50"]) & (data["SMA20"].shift(1) <= data["SMA50"].shift(1))
data["CrossDown"] = (data["SMA20"] < data["SMA50"]) & (data["SMA20"].shift(1) >= data["SMA50"].shift(1))

# ==============================
# Estrategia: Position (0/1)
# ==============================
data["Position"] = np.nan
data.loc[data["CrossUp"], "Position"] = 1
data.loc[data["CrossDown"], "Position"] = 0

# Mantener última posición hasta el próximo cruce
data["Position"] = data["Position"].ffill().fillna(0).astype(int)

# ==============================
# Retornos (sin trampa) + equity
# ==============================
data["BuyHoldReturn"] = data["DailyReturn"]

# Usamos la posición del día anterior (evita look-ahead)
data["StrategyReturn"] = data["Position"].shift(1) * data["DailyReturn"]

data["BuyHoldEquity"] = (1 + data["BuyHoldReturn"].fillna(0)).cumprod()
data["StrategyEquity"] = (1 + data["StrategyReturn"].fillna(0)).cumprod()

# ==============================
# Costos de transacción (net)
# ==============================
data["Trade"] = data["Position"].diff().abs().fillna(0)  # 1 cuando cambia 0->1 o 1->0
data["StrategyReturnNet"] = data["StrategyReturn"].fillna(0) - data["Trade"] * transaction_cost
data["StrategyEquityNet"] = (1 + data["StrategyReturnNet"]).cumprod()

# ==============================
# Métricas (volatilidad, sharpe, drawdown)
# ==============================
def safe_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    std = r.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return (r.mean() / std) * np.sqrt(252)

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    return float(drawdown.min())

strategy_vol = float(data["StrategyReturn"].dropna().std() * np.sqrt(252))
buyhold_vol = float(data["BuyHoldReturn"].dropna().std() * np.sqrt(252))

strategy_sharpe = safe_sharpe(data["StrategyReturn"])
buyhold_sharpe = safe_sharpe(data["BuyHoldReturn"])

strategy_dd = max_drawdown(data["StrategyEquity"])
strategy_dd_net = max_drawdown(data["StrategyEquityNet"])
buyhold_dd = max_drawdown(data["BuyHoldEquity"])

strategy_total = float(data["StrategyEquity"].iloc[-1] - 1)
strategy_total_net = float(data["StrategyEquityNet"].iloc[-1] - 1)
buyhold_total = float(data["BuyHoldEquity"].iloc[-1] - 1)

# ==============================
# Gráfico
# ==============================
plt.figure()
plt.plot(data["Date"], data["BuyHoldEquity"], label="Buy & Hold")
plt.plot(data["Date"], data["StrategyEquity"], label="Strategy (MA Cross)")
plt.plot(data["Date"], data["StrategyEquityNet"], label="Strategy (MA Cross, net)")
plt.title(f"{ticker} - Equity Curves ({period})")
plt.xlabel("Date")
plt.ylabel("Equity (growth of $1)")
plt.legend()
plt.tight_layout()

plot_path = out_dir / f"{ticker}_{period}_equity.png"
plt.savefig(plot_path, dpi=150)
plt.close()

# ==============================
# Output en consola (esencial)
# ==============================
print("\n===== DATA SUMMARY =====")
print("Ticker:", ticker, "| Period:", period)
print("Shape:", data.shape)
print("Golden Cross:", int(data["CrossUp"].sum()), "| Death Cross:", int(data["CrossDown"].sum()))
print("Position counts:\n", data["Position"].value_counts(dropna=False))

print("\n===== PERFORMANCE SUMMARY =====")
print(f"Strategy total return (gross): {strategy_total*100:.2f}%")
print(f"Strategy total return (net):   {strategy_total_net*100:.2f}% (cost={transaction_cost*100:.2f}%/trade)")
print(f"Buy & Hold total return:       {buyhold_total*100:.2f}%")
print(f"Equity plot saved to: {plot_path}")

print("\n===== RISK METRICS =====")
print(f"Volatility (ann) - Strategy: {strategy_vol:.2%} | Buy&Hold: {buyhold_vol:.2%}")
print(f"Sharpe - Strategy: {strategy_sharpe:.2f} | Buy&Hold: {buyhold_sharpe:.2f}")
print(f"Max Drawdown - Strategy(gross): {strategy_dd:.2%} | Strategy(net): {strategy_dd_net:.2%} | Buy&Hold: {buyhold_dd:.2%}")

# ==============================
# Guardar CSV
# ==============================
csv_path = out_dir / f"{ticker}_{period}_clean.csv"
data.to_csv(csv_path, index=False)
print(f"\nData saved to: {csv_path}")

