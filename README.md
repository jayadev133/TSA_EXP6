## Devloped by: JAYADEV PALLINTI
## Register Number: 212223240058
## Date: 06-10-2025

# Ex.No: 6                   HOLT WINTERS METHOD

### AIM:
To implement the Holt Winters Method Model using Python.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as datetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicative trend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and Evaluate the model  predictions against test data
6. Create teh final model and predict future data and plot it

### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose

# Load data
data = pd.read_excel('Coffe_sales.xlsx')
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')

sales_monthly = data.groupby(pd.Grouper(key='date', freq='MS'))['money'].sum().dropna()

# 1. Plot original time series
plt.figure(figsize=(12, 6))
sales_monthly.plot()
plt.title("Monthly Coffee Sales Over Time")
plt.ylabel("Monthly Sales Amount")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.savefig("plot1_sales_time_series.jpg")
plt.show()   # <--- Display in Colab
plt.close()

# 2. Scaled data
scaler = MinMaxScaler()
scaled_sales = scaler.fit_transform(sales_monthly.values.reshape(-1, 1))
scaled_series = pd.Series(scaled_sales.flatten(), index=sales_monthly.index)
plt.figure(figsize=(12, 6))
scaled_series.plot()
plt.title("Scaled Monthly Coffee Sales")
plt.ylabel("Scaled Sales")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.savefig("plot2_scaled_sales.jpg")
plt.show()   # <--- Display in Colab
plt.close()

# 3. Time series decomposition
try:
    decomposition = seasonal_decompose(sales_monthly, model='additive', period=min(12, len(sales_monthly)//2))
    fig = decomposition.plot()
    plt.suptitle('Time Series Decomposition (Coffee Sales)')
    plt.tight_layout()
    plt.savefig('plot3_decomposition.jpg')
    plt.show()  # <--- Display in Colab
    plt.close()
except Exception as e:
    print(f"Decomposition error: {e}")

# 4. Train/test split and Holt-Winters
scaled_adj = scaled_series + 0.1   # for positive values
train_size = int(len(scaled_adj) * 0.8)
train, test = scaled_adj[:train_size], scaled_adj[train_size:]

seasonal_periods = 12 if len(sales_monthly) >= 24 else None
if seasonal_periods:
    model = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=seasonal_periods).fit()
else:
    model = ExponentialSmoothing(train, trend="add", seasonal=None).fit()

test_pred = model.forecast(len(test))
plt.figure(figsize=(12, 6))
train.plot(label='Train')
test.plot(label='Test')
test_pred.plot(label='Forecast', linestyle='--')
plt.title("Holt-Winters Train/Test Split and Forecast")
plt.ylabel("Scaled Sales")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plot4_holt_winters_split.jpg")
plt.show()   # <--- Display in Colab
plt.close()

# 5. Final forecast (optional, if needed)
if seasonal_periods:
    final_model = ExponentialSmoothing(scaled_adj, trend="add", seasonal="mul", seasonal_periods=seasonal_periods).fit()
else:
    final_model = ExponentialSmoothing(scaled_adj, trend="add", seasonal=None).fit()
future_steps = 12
future_pred = final_model.forecast(future_steps)
future_idx = pd.date_range(start=scaled_adj.index[-1] + pd.DateOffset(months=1), periods=future_steps, freq="MS")
future_pred.index = future_idx

all_data = pd.Series(scaler.inverse_transform(scaled_adj.values.reshape(-1, 1)).flatten(), index=scaled_adj.index)
future_original = pd.Series(scaler.inverse_transform(future_pred.values.reshape(-1, 1)).flatten(), index=future_idx)

plt.figure(figsize=(14, 7))
all_data.plot(label="Historical Sales", linewidth=2)
future_original.plot(label="Future Forecast", linestyle="--", linewidth=2)
plt.title("Coffee Sales Holt-Winters Forecast")
plt.ylabel("Sales Amount")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plot5_final_forecast.jpg")
plt.show()   # <--- Display in Colab
plt.close()

```
### OUTPUT:
 
 Scaled_data plot:
<img width="1189" height="590" alt="TS EXP 6 P1" src="https://github.com/user-attachments/assets/c1eb94ef-29d7-415d-95c0-e8b6281e709f" />


Decomposed plot:

<img width="630" height="475" alt="TS EXP 6 P2" src="https://github.com/user-attachments/assets/074989ff-fb15-4243-b8b4-fd46dbba8795" />


Test prediction:

<img width="1389" height="690" alt="TS EXP 6 P3" src="https://github.com/user-attachments/assets/ffe60b5f-5994-4256-89c9-fecec9a207ee" />


Model performance metrics:

RMSE:


Standard deviation and mean:


Final prediction:

<img width="1189" height="590" alt="TS EXP 6 P4" src="https://github.com/user-attachments/assets/ac8b9492-573e-46c8-8aeb-32e8c7965af3" />


<img width="1189" height="590" alt="TS EXP 6 P5" src="https://github.com/user-attachments/assets/8b32fe6d-12cd-4b3e-bc1d-e2f1b146a005" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
