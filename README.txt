# Sales Forecasting Using ARIMA

## Files
- `sales_forecasting.py`: complete project code
- `sales_data.csv`: sample monthly sales dataset (January 2021 to December 2025)
- `output/`: generated graphs and result CSV files

## Run
1. Keep `sales_forecasting.py` and `sales_data.csv` in the same folder.
2. Open the folder in VS Code.
3. Open the terminal in that folder.
4. Run: `python sales_forecasting.py`

The script selects a simple ARIMA (p,d,q) model using AIC, evaluates it on a holdout test set, then refits on all data and forecasts the next 12 months.

Note: The supplied dataset is synthetic sample data for learning and demonstration. Replace it with real sales records for a real business forecast.
