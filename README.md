# 股票市场预测与自动化交易模拟

## 项目简介
一个基于 Python 的股票价格预测和自动化交易项目，使用 SVR 模型预测 AAPL 股价，并实现简单的交易策略。

## 安装
1. 安装 Python 3.9.18。
2. 安装依赖：
   ```bash
   pip install pandas numpy yfinance scikit-learn matplotlib seaborn apscheduler
## 运行 
a) main.py 查看单次预测和交易结果：python main.py
b) 运行 scheduler.py 启动自动化任务：python scheduler.py

## 结果
收益率：6.52%
最大回撤：1.78%

## 资金曲线图：
![aapl_plot](https://github.com/user-attachments/assets/a3ed197c-9f85-4eea-9624-5f3b5f68f8a2)

## 预测结果图：
![svr_prediction](https://github.com/user-attachments/assets/3265ebd7-39ab-4930-a9dd-978acd4cda28)
