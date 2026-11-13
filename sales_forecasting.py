"""
Sales Forecasting Using ARIMA
Run:
    python sales_forecasting.py

Input:
    sales_data.csv (must be in the same folder as this script)

Outputs are saved in the output/ folder.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# -------------------- 1. FILE PATHS --------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "sales_data.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------- 2. LOAD DATA --------------------
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_FILE.name}. Keep it in the same folder as this script."
    )

data = pd.read_csv(DATA_FILE)

required_columns = {"Date", "Sales"}
if not required_columns.issubset(data.columns):
    raise ValueError("CSV must contain exactly these required columns: Date and Sales.")

data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
data["Sales"] = pd.to_numeric(data["Sales"], errors="coerce")
data = data.dropna(subset=["Date", "Sales"])
data = data.drop_duplicates(subset=["Date"])
data = data.sort_values("Date")
data = data.set_index("Date")

# Convert to monthly frequency. For multiple records in a month, sum the sales.
series = data["Sales"].resample("MS").sum().asfreq("MS")
series = series.interpolate(method="time")

if len(series) < 12:
    raise ValueError("Please provide at least 12 months of sales data.")

print("\n========== SALES DATA ==========")
print(series.head())
print(f"\nTotal monthly records: {len(series)}")
print(f"Missing values after preprocessing: {series.isna().sum()}")

# -------------------- 3. PLOT HISTORICAL SALES --------------------
plt.figure(figsize=(12, 5))
plt.plot(series.index, series.values, marker="o", linewidth=1.8)
plt.title("Historical Monthly Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_historical_sales.png", dpi=150)
plt.show()

# -------------------- 4. TREND AND SEASONALITY --------------------
if len(series) >= 24:
    decomposition = seasonal_decompose(series, model="additive", period=12)
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_sales_decomposition.png", dpi=150)
    plt.show()

# -------------------- 5. STATIONARITY CHECK --------------------
adf_result = adfuller(series.dropna())
print("\n========== STATIONARITY TEST ==========")
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
if adf_result[1] < 0.05:
    print("The series appears stationary (p-value < 0.05).")
else:
    print("The series may be non-stationary; differencing may be required.")

# -------------------- 6. TRAIN-TEST SPLIT --------------------
test_size = max(3, int(round(len(series) * 0.20)))
train = series.iloc[:-test_size]
test = series.iloc[-test_size:]

print("\n========== TRAIN / TEST ==========")
print(f"Training records: {len(train)}")
print(f"Testing records: {len(test)}")

# -------------------- 7. SELECT ARIMA ORDER BY AIC --------------------
# A small grid keeps the example practical and avoids extra packages.
best_aic = float("inf")
best_order = None
best_model = None

for p in range(0, 4):
    for d in range(0, 3):
        for q in range(0, 4):
            try:
                candidate = ARIMA(
                    train,
                    order=(p, d, q),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit()
                if np.isfinite(candidate.aic) and candidate.aic < best_aic:
                    best_aic = candidate.aic
                    best_order = (p, d, q)
                    best_model = candidate
            except Exception:
                continue

if best_model is None:
    raise RuntimeError("ARIMA could not be fitted. Check your data and try again.")

print("\n========== BEST ARIMA MODEL ==========")
print(f"Selected order (p, d, q): {best_order}")
print(f"AIC: {best_aic:.2f}")
print(best_model.summary())

# -------------------- 8. TEST FORECAST --------------------
test_forecast = best_model.forecast(steps=len(test))
test_forecast = pd.Series(np.asarray(test_forecast), index=test.index)

mae = mean_absolute_error(test, test_forecast)
mse = mean_squared_error(test, test_forecast)
rmse = np.sqrt(mse)

non_zero_actual = test.replace(0, np.nan)
mape = (np.abs((test - test_forecast) / non_zero_actual)).mean() * 100

print("\n========== MODEL EVALUATION ==========")
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Save actual vs predicted test values
comparison = pd.DataFrame({
    "Actual_Sales": test,
    "Predicted_Sales": test_forecast
})
comparison.to_csv(OUTPUT_DIR / "test_predictions.csv")

plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label="Training sales")
plt.plot(test.index, test, marker="o", label="Actual test sales")
plt.plot(test_forecast.index, test_forecast, marker="o", linestyle="--",
         label="Predicted test sales")
plt.title("Actual vs Predicted Sales (Test Period)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_actual_vs_predicted.png", dpi=150)
plt.show()

# -------------------- 9. REFIT ON ALL DATA AND FORECAST --------------------
final_model = ARIMA(
    series,
    order=best_order,
    enforce_stationarity=False,
    enforce_invertibility=False
).fit()

FORECAST_MONTHS = 12
future_result = final_model.get_forecast(steps=FORECAST_MONTHS)
future_mean = future_result.predicted_mean
confidence = future_result.conf_int()

future_dates = pd.date_range(
    start=series.index[-1] + pd.offsets.MonthBegin(1),
    periods=FORECAST_MONTHS,
    freq="MS"
)
future_mean.index = future_dates
confidence.index = future_dates

forecast_table = pd.DataFrame({
    "Forecast_Sales": future_mean,
    "Lower_Confidence_Bound": confidence.iloc[:, 0],
    "Upper_Confidence_Bound": confidence.iloc[:, 1]
})
forecast_table.to_csv(OUTPUT_DIR / "future_sales_forecast.csv")

print("\n========== NEXT 12 MONTHS SALES FORECAST ==========")
print(forecast_table.round(2))

plt.figure(figsize=(12, 5))
plt.plot(series.index, series, label="Historical sales")
plt.plot(future_mean.index, future_mean, marker="o", linestyle="--",
         label="Forecast sales")
plt.fill_between(
    future_dates,
    confidence.iloc[:, 0].values,
    confidence.iloc[:, 1].values,
    alpha=0.2,
    label="95% confidence interval"
)
plt.title("Sales Forecast for the Next 12 Months (ARIMA)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_future_sales_forecast.png", dpi=150)
plt.show()

# -------------------- 10. SAVE MODEL METRICS --------------------
metrics = pd.DataFrame([{
    "ARIMA_Order": str(best_order),
    "AIC": best_aic,
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "MAPE_Percent": mape
}])
metrics.to_csv(OUTPUT_DIR / "model_evaluation.csv", index=False)

print("\nProject completed successfully!")
print(f"All graphs and result CSV files are saved in: {OUTPUT_DIR}")
