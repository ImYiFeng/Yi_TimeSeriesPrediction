import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import TensorBoard
import os
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# 加载训练数据
train_data_path = './data/china_gdp_train.csv'
train_data = pd.read_csv(train_data_path)

# 数据标准化/归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_gdp = scaler.fit_transform(train_data['GDP'].values.reshape(-1, 1))

# LSTM需要3D数据格式 [samples, timesteps, features]
X_train = scaled_gdp[:-1]  # 所有数据除了最后一个作为特征
y_train = scaled_gdp[1:]   # 所有数据除了第一个作为标签

# 为了使X_train符合3D输入，我们增加一个维度
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanAbsoluteError()])

# 早停法来防止过拟合
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# 训练模型
history = model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, tensorboard_callback],
    verbose=2
)

# 保存模型
model.save('./models/lstm_model.h5')

print("模型训练完成并已保存为 lstm_model.h5")
