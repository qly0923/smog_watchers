import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('imputed_data.csv', parse_dates=['id'], index_col='id')
selected_columns = ['valeur_PM25']
data = data[selected_columns].resample('D').mean()
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]
# 定义SARIMA模型
model = SARIMAX(train_data['valeur_PM25'], order=(1, 0, 1), seasonal_order=(1, 1, 1, 365))
sarima_model = model.fit()

# 使用模型进行预测
forecast = sarima_model.get_forecast(steps=len(test_data))
forecast_index = test_data.index
forecast_values = forecast.predicted_mean


# 计算 MAE
mae = mean_absolute_error(test_data, forecast_values)
print(f'Mean Absolute Error (MAE): {mae}')