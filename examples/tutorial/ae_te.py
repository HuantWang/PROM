import numpy as np

# 创建原始数组
array = np.array([1, 2, 3, 10])

# 转换为 float64 类型并立即舍入到五位小数
array = np.round(array.astype(np.float64), 5)

# 验证数组内容
print("Array stored with five decimal places:", array)
print("Data type of array:", array.dtype)
