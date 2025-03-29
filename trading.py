import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')# 设置非GUI后端

def run_trading_strategy(y_test_original, predictions_original):
    initial_cash = 10000
    cash = initial_cash
    shares = 0
    portfolio_values = [initial_cash]
    buy_price = 0  # 记录买入价格
    stop_loss = 0.05  # 5%止损

    for i in range(len(predictions_original) - 1):
        current_price = float(y_test_original[i])
        predicted_price = float(predictions_original[i + 1])
        if predicted_price > current_price and cash >= current_price:
            shares_to_buy = int(cash // current_price)
            shares += shares_to_buy
            cash -= shares_to_buy * current_price
            buy_price = current_price  # 记录买入价格
        elif shares > 0:
            # 止损检查
            if current_price < buy_price * (1 - stop_loss):
                cash += shares * current_price
                shares = 0
            elif predicted_price < current_price:
                cash += shares * current_price
                shares = 0
        portfolio_values.append(cash + shares * current_price)

    final_value = cash + shares * float(y_test_original[-1])
    print("最终价值:", final_value)
    print("收益率:", (final_value - initial_cash) / initial_cash * 100, "%")

    # 计算最大回撤
    portfolio_values = np.array(portfolio_values)
    drawdowns = []
    for i in range(len(portfolio_values)):
        peak = np.max(portfolio_values[:i + 1])
        drawdown = (peak - portfolio_values[i]) / peak
        drawdowns.append(drawdown)
    max_drawdown = np.max(drawdowns)
    print("最大回撤:", max_drawdown * 100, "%")

    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('portfolio_value.png')
    # plt.show()
    plt.close()
    return final_value, portfolio_values