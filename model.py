from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)# 将数据分成训练集（80%）和测试集（20%），不打乱数据
    # 使用径向基函数rbf（Radial Basis Function），
    # C增大惩罚参数，让模型更严格地减少误差(过大会过拟合)
    # gamma调整RBF核的宽度，控制模型对数据的拟合程度(越大越关注局部)
    model = SVR(kernel='rbf', C=100, gamma=0.1)
    model.fit(X_train, y_train)# 模型学习X_train（60天价格）和y_train（第61天价格）之间的关系
    predictions = model.predict(X_test)
    print("X_train 形状:", X_train.shape)
    print("X_test 形状:", X_test.shape)
    return model, X_train, X_test, y_train, y_test, predictions
    