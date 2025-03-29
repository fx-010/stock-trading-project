import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def fetch_data(stock='AAPL', start='2020-01-01', end='2025-01-01'):
    data = yf.download(stock, start=start, end=end)
    data.to_csv('aapl_data.csv')
    print("数据已保存到 aapl_data.csv")
    print("缺失值检查：", data.isnull().sum())
    return data

# 数据预处理
def preprocess_data(data, time_step=60):# 60天作为时间窗口
    close_data = data[['Close']]# data[['Close']]返回一个DataFrame
    scaler = MinMaxScaler()# 归一化
    scaled_data = scaler.fit_transform(close_data)# 需要二维输入
    print("归一化后的前 5 个值：", scaled_data[:5])
    
    # 将时间序列数据转化为监督学习格式
    def create_dataset(data, time_step):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), 0])# 取第i到i+time_step-1天的价格（共60天）
            y.append(data[i + time_step, 0])# 取第i+time_step天的价格作为目标
        return np.array(X), np.array(y)# 将X和y从列表转为numpy数组，方便模型训练

    X, y = create_dataset(scaled_data, time_step)
    np.save('X.npy', X)
    np.save('y.npy', y)
    print("X 形状：", X.shape)
    print("y 形状：", y.shape)
    return X, y, scaler, close_data
