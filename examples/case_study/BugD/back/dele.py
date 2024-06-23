# import os
#
# # 定义根文件夹路径
# root_folder = "/home/huanting/model/bug_detect/CodeXGLUE-main/Defect-detection/data"
#
# # 定义连续两行的条件
# line1 = "-----label-----\n"
# line2 = "0\n"
#
# # 遍历文件夹及其子文件夹中的 TXT 文件
# for folder_path, _, file_names in os.walk(root_folder):
#     for file_name in file_names:
#         if file_name.endswith(".txt"):
#             file_path = os.path.join(folder_path, file_name)
#
#             # 读取文件内容
#             with open(file_path, "r") as file:
#                 lines = file.readlines()
#
#             # 检查连续两行的条件
#             if line1 in lines and line2 in lines[lines.index(line1) + 1]:
#                 # 删除文件
#                 os.remove(file_path)
#                 print(f"Deleted file: {file_path}")


import os

# 定义根文件夹路径
root_folder = "/home/huanting/model/bug_detect/CodeXGLUE-main/Defect-detection/data"

# 遍历文件夹
for folder_path, _, file_names in os.walk(root_folder):
    txt_count = 0
    for file_name in file_names:
        if file_name.endswith(".txt"):
            txt_count += 1

    # 输出每个文件夹中的 TXT 文件数量
    print(f"Folder: {folder_path}")
    print(f"Number of TXT files: {txt_count}")
    print()
