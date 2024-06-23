import json
import matplotlib.pyplot as plt
import os

def find_constant_parts(data):
    start_index = 0
    end_index = len(data) - 1

    # 查找前面一部分不变的数据
    for i in range(1, len(data)):
        if data[i] != data[0]:
            start_index = i - 2
            break

    # 查找后面一部分不变的数据
    for i in range(len(data) - 2, -1, -1):
        if data[i] != data[-1]:
            end_index = i + 2
            break

    return start_index, end_index


# 指定包含JSON文件的文件夹路径
folder_path = "/home/huanting/model/compy-learn-master/dict"

# 获取文件夹中所有JSON文件的文件名列表
json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

# 遍历每个JSON文件
for json_file in json_files:
    file_path = os.path.join(folder_path, json_file)

    with open(file_path, 'r') as file:
        record = json.load(file)
    print(file)

    # 循环遍历每个方法并绘制单独的图表
    # methods = ['naive', 'score', 'cumulated_score', 'random_cumulated_score', 'top_k','mixture']
    colors = ['b', 'g', 'r', 'c', 'm']  # 可以自定义颜色
    methods = ['score']
    for i, method in enumerate(methods):
        pre_data = record[method]['pre']
        rec_data = record[method]['rec']
        f1_data = record[method]['f1']

        # start_index, end_index = find_constant_parts(f1_data)
        # f1_data = f1_data[start_index:end_index + 1]
        # pre_data = pre_data[start_index:end_index + 1]
        # rec_data = rec_data[start_index:end_index + 1]

        # 创建新的图表
        plt.figure(figsize=(10, 5))

        # 绘制预测精度（pre）的趋势图
        plt.plot(pre_data, label='Precision', color=colors[0], linestyle='-')

        # 绘制召回率（rec）的趋势图
        plt.plot(rec_data, label='Recall', color=colors[1], linestyle='--')

        # 绘制F1分数（f1）的趋势图
        plt.plot(f1_data, label='F1 Score', color=colors[2], linestyle=':')

        plt.xlabel(json_file)
        plt.ylabel('值')
        plt.title(f'{method} 方法指标趋势图')

        # 显示图例
        plt.legend()

        # 显示图形
        plt.show()
        print(" ")