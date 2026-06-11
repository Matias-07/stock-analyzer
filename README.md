# stock-analyzer
 
Backtesting de una estrategia de cruce de medias móviles (SMA) sobre datos reales del mercado, con métricas de riesgo y costos de transacción. Escrito en Python con pandas, NumPy y matplotlib.
 
## Qué hace
 
Descarga datos históricos de una acción, calcula indicadores técnicos, simula una estrategia de trading basada en el cruce de medias móviles y la compara contra una estrategia pasiva de *buy & hold*, reportando rendimiento y riesgo.
 
- **Datos**: descarga precios históricos vía [yfinance](https://github.com/ranaroussi/yfinance).
- **Indicadores**: retornos diarios, volatilidad móvil (20 días), medias móviles SMA20 y SMA50.
- **Estrategia**: señal de compra/venta en los cruces de SMA20 y SMA50 (*golden cross* / *death cross*).
- **Backtest honesto**: usa la posición del día anterior (`Position.shift(1)`) para evitar *look-ahead bias*, e incluye costos de transacción configurables por operación.
- **Métricas de riesgo**: volatilidad anualizada, ratio de Sharpe y máximo *drawdown*, tanto para la estrategia como para *buy & hold*.
- **Salida**: gráfico de las curvas de capital (gross, net y buy & hold) y un CSV con todos los datos procesados.

## Cómo correrlo
 
```bash
# Clonar el repo
git clone https://github.com/Matias-07/stock-analyzer.git
cd stock-analyzer
 
# Crear entorno virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
 
# Ejecutar
python src/download_data.py
```
 
Los resultados (gráfico PNG y CSV) se generan en la carpeta `outputs/`.
 
## Configuración
 
Los parámetros se ajustan al principio de `src/download_data.py`:
 
| Parámetro | Descripción | Valor por defecto |
|---|---|---|
| `ticker` | Símbolo de la acción a analizar | `"AAPL"` |
| `period` | Período de datos históricos | `"6mo"` |
| `transaction_cost` | Costo por operación (entrada o salida) | `0.001` (0.1%) |
 
## Stack
 
`Python` · `pandas` · `NumPy` · `matplotlib` · `yfinance`
 
---
