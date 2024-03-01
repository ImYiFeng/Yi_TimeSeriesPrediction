import pandas as pd
import matplotlib.pyplot as plt

file_path = './data/china_gdp_preprocessed.csv'
data = pd.read_csv(file_path)

# 绘制时间序列图
plt.figure(figsize=(12, 6))
plt.plot(data['Year'], data['GDP'], marker='o', linestyle='-', color='blue')
plt.title('China GDP Over Time')
plt.xlabel('Year')
plt.ylabel('GDP (in RMB)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
