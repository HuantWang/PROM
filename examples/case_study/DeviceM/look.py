import json
import matplotlib.pyplot as plt
import os

def find_constant_parts(data):
    start_index = 0
    end_index = len(data) - 1

    # 
    for i in range(1, len(data)):
        if data[i] != data[0]:
            start_index = i - 2
            break

    # 
    for i in range(len(data) - 2, -1, -1):
        if data[i] != data[-1]:
            end_index = i + 2
            break

    return start_index, end_index


# JSON
folder_path = "/home/huanting/model/compy-learn-master/dict"

# JSON
json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

# JSON
for json_file in json_files:
    file_path = os.path.join(folder_path, json_file)

    with open(file_path, 'r') as file:
        record = json.load(file)
    print(file)

    # 
    # methods = ['naive', 'score', 'cumulated_score', 'random_cumulated_score', 'top_k','mixture']
    colors = ['b', 'g', 'r', 'c', 'm']  # 
    methods = ['score']
    for i, method in enumerate(methods):
        pre_data = record[method]['pre']
        rec_data = record[method]['rec']
        f1_data = record[method]['f1']

        # start_index, end_index = find_constant_parts(f1_data)
        # f1_data = f1_data[start_index:end_index + 1]
        # pre_data = pre_data[start_index:end_index + 1]
        # rec_data = rec_data[start_index:end_index + 1]

        # 
        plt.figure(figsize=(10, 5))

        # pre
        plt.plot(pre_data, label='Precision', color=colors[0], linestyle='-')

        # rec
        plt.plot(rec_data, label='Recall', color=colors[1], linestyle='--')

        # F1f1
        plt.plot(f1_data, label='F1 Score', color=colors[2], linestyle=':')

        plt.xlabel(json_file)
        plt.ylabel('')
        plt.title(f'{method} ')

        # 
        plt.legend()

        # 
        plt.show()
        print(" ")