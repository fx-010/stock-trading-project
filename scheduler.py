from apscheduler.schedulers.blocking import BlockingScheduler
from data_fetch import fetch_data, preprocess_data
from model import train_model
from trading import run_trading_strategy

def job():
    print("运行自动化任务...")
    # 数据获取和预处理
    data = fetch_data(start='2020-01-01', end='2025-01-01')
    X, y, scaler, close_data = preprocess_data(data)
    # 模型训练和预测
    model, X_train, X_test, y_train, y_test, predictions = train_model(X, y)
    predictions_reshaped = predictions.reshape(-1, 1)
    y_test_reshaped = y_test.reshape(-1, 1)
    predictions_original = scaler.inverse_transform(predictions_reshaped).flatten()
    y_test_original = scaler.inverse_transform(y_test_reshaped).flatten()
    # 交易策略
    final_value, portfolio_values = run_trading_strategy(y_test_original, predictions_original)

scheduler = BlockingScheduler()
scheduler.add_job(job, 'interval', minutes=1)  # 每分钟运行一次
scheduler.start()
