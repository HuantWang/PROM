import os
import tqdm
import random
import json
import shutil
import os


def findAllFile(dir):
    for root, ds, fs in os.walk(dir):
        for f in fs:
            yield root, f


# split to pos/neg
# file='/home/huanting/model/bug_detect/CodeXGLUE-main/Defect-detection/data/CWE-772'
# cwe = 'CWE-772'
#
# good_folder = '/home/huanting/model/bug_detect/CodeXGLUE-main/Defect-detection/data/data_gb/'+cwe+'/GOOD'
# bad_folder = '/home/huanting/model/bug_detect/CodeXGLUE-main/Defect-detection/data/data_gb/'+cwe+'/BAD'
#
# # 如果目标文件夹不存在，则创建它
# if not os.path.exists(good_folder):
#     os.makedirs(good_folder)
# if not os.path.exists(bad_folder):
#     os.makedirs(bad_folder)
#
# for root, file in findAllFile(file):
#     if file.endswith(".txt"):
#         with open(root + '/' + file) as f:
#             for line in f:
#                 if line == "-----label-----\n":
#                     flag = "label"
#                     continue
#                 if flag == "label":
#                     y = line.split()[0]
#                     if int(y) == int(0):
#                         shutil.copy(os.path.join(root, file), os.path.join(good_folder, file))
#                         break
#                     if int(y) == int(1):
#                         shutil.copy(os.path.join(root, file), os.path.join(bad_folder, file))
#                         break
#                     else:
#                         break

# file='/home/huanting/model/bug_detect/CodeXGLUE-main/Defect-detection/data/data_gb/CWE-190/GOOD'
# num=0
# for root, file in findAllFile(file):
#     if file.endswith(".txt"):
#         num+=1
#         if num>500:
#             os.remove(root+'/'+file)

import os
import random


def pre(
    folder_path="/home/huanting/model/bug_detect/CodeXGLUE-main/Defect-detection/data_u",
    random_seed=1234,
    num_folders=8,
):
    # 设置文件夹路径和目标数量
    # folder_path = "/home/huanting/model/bug_detect/CodeXGLUE-main/Defect-detection/data"  # 文件夹的路径

    num_files_per_folder = 100  # 每个文件夹中需要选择的文本文件数量

    # 获取文件夹列表
    folders = [
        folder
        for folder in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, folder))
    ]

    random.seed(random_seed)
    # 随机选择文件夹
    selected_folders = random.sample(folders, num_folders)

    # 创建数组用于存储选中的文件路径
    selected_files = []

    # 遍历选中的文件夹
    for folder in selected_folders:
        folder_dir = os.path.join(folder_path, folder)
        for root, dirs, files in os.walk(folder_dir):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    selected_files.append(file_path)

                    if len(selected_files) % num_files_per_folder == 0:
                        break
            if len(selected_files) == num_folders * num_files_per_folder:
                break

        if len(selected_files) == num_folders * num_files_per_folder:
            break
    # 输出选中的文件路径
    # print(selected_files)

    random.shuffle(selected_files)
    # file_name = []
    if os.path.exists(
        "/home/huanting/model/bug_detect/CodeXGLUE-main/Defect-detection/data/train.jsonl"
    ):
        try:
            os.remove(
                "/home/huanting/model/bug_detect/CodeXGLUE-main/Defect-detection/data/train.jsonl"
            )
            os.remove(
                "/home/huanting/model/bug_detect/CodeXGLUE-main/Defect-detection/data/valid.jsonl"
            )
            os.remove(
                "/home/huanting/model/bug_detect/CodeXGLUE-main/Defect-detection/data/test.jsonl"
            )
        except:
            pass

    # for root, file in findAllFile(selected_files):
    #     if file.endswith(".txt"):
    #         name = root + '/' + file
    #         file_name.append(name)

    for i in range(len(selected_files)):
        if i < len(selected_files) * 0.6:
            with open(
                "/home/huanting/model/bug_detect/CodeXGLUE-main/Defect-detection/data/train.jsonl",
                "a",
            ) as f:
                f.write(json.dumps(selected_files[i]) + "\n")
        if i >= len(selected_files) * 0.6 and i < len(selected_files) * 0.8:
            with open(
                "/home/huanting/model/bug_detect/CodeXGLUE-main/Defect-detection/data/valid.jsonl",
                "a",
            ) as f:
                f.write(json.dumps(selected_files[i]) + "\n")
        if i >= len(selected_files) * 0.8:
            with open(
                "/home/huanting/model/bug_detect/CodeXGLUE-main/Defect-detection/data/test.jsonl",
                "a",
            ) as f:
                f.write(json.dumps(selected_files[i]) + "\n")

    print("data preprocess finish")


# pre()

import os

# 定义文件夹路径


# 递归遍历文件夹及其子文件夹中的 TXT 文件
def process_folder(folder_path):
    all_part = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if os.path.isdir(file_path):
            process_folder(file_path)
        elif file_name.endswith(".txt"):
            # 读取文件内容
            with open(file_path, "r") as file:
                content = file.read()

            # 提取以 "CWE" 开头的字符串
            lines = content.splitlines()
            cwe_strings = [line for line in lines if line.startswith("CWE")]
            try:
                parts = cwe_strings[0].split("_")[0]
            except:
                continue
            # 输出提取的字符串
            all_part.append(parts)
    all_part = list(set(all_part))
    print()


import os


def process_folder(folder_path, count_dict):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if os.path.isdir(file_path):
            process_folder(file_path, count_dict)
        elif file_name.endswith(".txt"):
            # 读取文件内容
            with open(file_path, "r") as file:
                content = file.read()

            # 提取以 "CWE" 开头的字符串
            lines = content.splitlines()
            cwe_strings = [line for line in lines if line.startswith("CWE")]
            if cwe_strings:
                parts = cwe_strings[0].split("_")[0]

                # 更新字典中字符串出现的次数
                count_dict[parts] = count_dict.get(parts, 0) + 1


# # 文件夹路径
# folder_path = "/home/huanting/model/bug_detect/CodeXGLUE-main/Defect-detection/data/CWE-665"
# # 字典用于记录字符串出现的次数
# count_dict = {}
# # 处理文件夹及其子文件夹中的 TXT 文件
# process_folder(folder_path, count_dict)
# # 输出统计结果
# for parts, count in count_dict.items():
#     print(f"字符串 '{parts}' 出现了 {count} 次")


import os
import shutil


def copy_files_with_prefix(source_folder, destination_folder, prefix):
    # 遍历源文件夹及其子文件夹中的文件
    for root, dirs, files in os.walk(source_folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # 读取文件内容
            with open(file_path, "r") as file:
                content = file.read()

            # 检查文件内容是否以指定前缀开头
            if prefix in content:
                destination_file = os.path.join(destination_folder, file_name)

                # 复制文件到目标文件夹
                shutil.copy2(file_path, destination_file)


# # 源文件夹路径
# source_folder = "/home/huanting/model/bug_detect/CodeXGLUE-main/Defect-detection/data_c/CWE-20"
# # 目标文件夹路径
# destination_folder = "/home/huanting/model/bug_detect/CodeXGLUE-main/Defect-detection/data_u/CWE-122"
#
# # 以 "CWE123" 开头的文件复制到目标文件夹
# os.makedirs(destination_folder, exist_ok=True)
# copy_files_with_prefix(source_folder, destination_folder, "CWE122")

""" 
20:['CWE126', 'CWE78', 'CWE121', 'CWE134', 'CWE127', 'CWE789', 'CWE416', 'CWE124', 'CWE122', 'CWE190', 'CWE415']
字符串 'CWE78' 出现了 1603 次
字符串 'CWE127' 出现了 96 次
字符串 'CWE190' 出现了 710 次
字符串 'CWE122' 出现了 671 次
字符串 'CWE789' 出现了 159 次
字符串 'CWE124' 出现了 94 次
字符串 'CWE134' 出现了 73 次
字符串 'CWE121' 出现了 196 次
字符串 'CWE126' 出现了 55 次
字符串 'CWE415' 出现了 49 次
字符串 'CWE416' 出现了 41 次
74 ['CWE134', 'CWE90', 'CWE464', 'CWE78']
字符串 'CWE134' 出现了 420 次
字符串 'CWE78' 出现了 711 次
字符串 'CWE90' 出现了 17 次
字符串 'CWE464' 出现了 14 次
77 ['CWE78']
字符串 'CWE78' 出现了 2144 次
78 ['CWE78']
字符串 'CWE78' 出现了 2120 次
190 ['CWE190']
字符串 'CWE190' 出现了 5130 次

400 ['CWE789', 'CWE400', 'CWE401', 'CWE775', 'CWE773']
字符串 'CWE401' 出现了 703 次
字符串 'CWE400' 出现了 504 次
字符串 'CWE789' 出现了 351 次
字符串 'CWE775' 出现了 87 次
字符串 'CWE773' 出现了 107 次

665 ['CWE665', 'CWE789', 'CWE457']
字符串 'CWE457' 出现了 309 次
字符串 'CWE665' 出现了 126 次
字符串 'CWE789' 出现了 279 次

668 ['CWE615', 'CWE226', 'CWE535', 'CWE256', 'CWE427', 'CWE377', 'CWE15', 'CWE244', 'CWE526', 'CWE534']
字符串 'CWE377' 出现了 144 次
字符串 'CWE427' 出现了 577 次
字符串 'CWE256' 出现了 112 次
字符串 'CWE526' 出现了 18 次
字符串 'CWE534' 出现了 36 次
字符串 'CWE226' 出现了 72 次
字符串 'CWE615' 出现了 18 次
字符串 'CWE15' 出现了 58 次
字符串 'CWE244' 出现了 69 次
字符串 'CWE535' 出现了 36 次

704 ['CWE197', 'CWE196', 'CWE588', 'CWE194', 'CWE195', 'CWE843', 'CWE681']
字符串 'CWE197' 出现了 1044 次
字符串 'CWE681' 出现了 54 次
字符串 'CWE195' 出现了 1392 次
字符串 'CWE194' 出现了 1392 次
字符串 'CWE843' 出现了 104 次
字符串 'CWE588' 出现了 52 次
字符串 'CWE196' 出现了 18 次

772 
字符串 'CWE401' 出现了 1148 次
字符串 'CWE775' 出现了 156 次
"""
