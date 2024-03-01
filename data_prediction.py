import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# 加载训练和测试数据集
train_data_path = './data/china_gdp_train.csv'
test_data_path = './data/china_gdp_preprocessed.csv'
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# 加载模型
model = load_model('./models/lstm_model.h5')

# 数据标准化/归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data['GDP'].values.reshape(-1, 1))  # 找到归一化参数

# 仅对测试数据进行归一化
scaled_gdp_test = scaler.transform(test_data['GDP'].values.reshape(-1, 1))

# 为了使X_test符合3D输入，我们增加一个维度
X_test = scaled_gdp_test[:-1]
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# 使用模型进行预测
predicted_scaled_gdp = model.predict(X_test)

# 逆归一化预测结果
predicted_gdp = scaler.inverse_transform(predicted_scaled_gdp)

# 绘制实际GDP值和预测GDP值的对比图
plt.figure(figsize=(12, 6))
plt.plot(test_data['Year'][1:], test_data['GDP'][1:], label='Actual GDP', marker='o')
plt.plot(test_data['Year'][1:], predicted_gdp, label='Predicted GDP', marker='o', linestyle='--')
plt.title('Actual vs Predicted GDP')
plt.xlabel('Year')
plt.ylabel('GDP (in RMB)')
plt.legend()
plt.show()

# 打印实际值和预测值
for actual, predicted, year in zip(test_data['GDP'][1:], predicted_gdp, test_data['Year'][1:]):
    print(f"Year: {year}, Actual GDP: {actual}, Predicted GDP: {predicted[0]}")
