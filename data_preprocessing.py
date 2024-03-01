import pandas as pd

file_path = './data/china_gdp.csv'
data = pd.read_csv(file_path)

# data['Year'] = pd.to_datetime(data['Year'], format='%Y')

print(data.head())

preprocessed_file_path = './data/china_gdp_preprocessed.csv'
data.to_csv(preprocessed_file_path, index=False)

# 这里调整分割数据的位置
split_point = int(len(data) * 0.8)
train_data = data[:split_point]
test_data = data[split_point:]

train_data_path = './data/china_gdp_train.csv'
test_data_path = './data/china_gdp_test.csv'
train_data.to_csv(train_data_path, index=False)
test_data.to_csv(test_data_path, index=False)

print(f"数据预处理完成，请检查数据格式")
print(f"预处理数据位置{train_data_path}")
print(f"训练数据位置{test_data_path}")
print(f"测试数据位置{test_data_path}")
