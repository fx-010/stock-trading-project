import pandas as pd
import matplotlib.pyplot as plt
from data_fetch import fetch_data, preprocess_data
from model import train_model
from trading import run_trading_strategy
import matplotlib
matplotlib.use('Agg')# 设置非GUI后端

# 数据获取和预处理
data = fetch_data()# 调用fetch_data()函数获取股票数据
X, y, scaler, close_data = preprocess_data(data)
print("数据统计：", close_data.describe())
plt.figure(figsize=(10, 6))
plt.plot(close_data, label='Close Price')
data['MA20'] = close_data.rolling(window=20).mean()# 计算20日移动平均线
plt.plot(data['MA20'], label='20-Day Moving Average', color='orange')
plt.title('AAPL Close Price and 20-Day MA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig('aapl_plot.png')
# plt.show()
plt.close()

# 模型训练和预测(训练好的模型，特征数据(X)，目标数据(y)，模型对X_test的预测结果)
model, X_train, X_test, y_train, y_test, predictions = train_model(X, y)# 调用train_model()函数，训练SVR模型
predictions_reshaped = predictions.reshape(-1, 1)# '-1': 自动计算行数；'1':变成1行
y_test_reshaped = y_test.reshape(-1, 1)
predictions_original = scaler.inverse_transform(predictions_reshaped).flatten()# 将标准化后的预测值（0-1范围）还原为原始价格;展平为一维
y_test_original = scaler.inverse_transform(y_test_reshaped).flatten()  # 展平为一维
plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label='Actual Price')
plt.plot(predictions_original, label='Predicted Price')
plt.title('SVR Prediction vs Actual')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig('svr_prediction.png')
# plt.show()
plt.close()

# 交易策略
final_value, portfolio_values = run_trading_strategy(y_test_original, predictions_original)
