import os
import tqdm
import random
import json
import shutil


def findAllFile(dir):
    """
        Recursively finds all files in a given directory.

        Parameters
        ----------
        dir : str
            The directory path to search for files.

        Yields
        ------
        tuple
            A tuple containing the root directory and the file name for each file found.
        """
    for root, ds, fs in os.walk(dir):
        for f in fs:
            yield root, f


def pre(
    dataset="../../../benchmark/Bug",
    random_seed=1234,
    num_folders=8,
):
    """
    Preprocesses files from the given dataset by selecting a sample of text files from each folder,
    shuffling, and splitting them into training, validation, and testing sets.

    Parameters
    ----------
    dataset : str, optional
        Path to the dataset directory, by default "../../../benchmark/Bug".
    random_seed : int, optional
        Seed for randomization, by default 1234.
    num_folders : int, optional
        Number of folders to sample from within the dataset, by default 8.

    Returns
    -------
    None
    """
    num_files_per_folder = 150
    folders = [
        folder
        for folder in os.listdir(dataset)
        if os.path.isdir(os.path.join(dataset, folder))
    ]

    random.seed(random_seed)
    selected_folders = random.sample(folders, num_folders)


    selected_files = []


    for folder in selected_folders:
        folder_dir = os.path.join(dataset, folder)
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

    # print(selected_files)

    random.shuffle(selected_files)
    # file_name = []
    if os.path.exists(dataset + "/train.jsonl"):
        try:

            os.remove(dataset + "/train.jsonl")
            os.remove(dataset + "/valid.jsonl")
            os.remove(dataset + "/test.jsonl")
        except:
            pass

    # for root, file in findAllFile(selected_files):
    #     if file.endswith(".txt"):
    #         name = root + '/' + file
    #         file_name.append(name)

    for i in range(len(selected_files)):
        if i < len(selected_files) * 0.6:
            with open(
                dataset + "/train.jsonl",
                "a",
            ) as f:
                f.write(json.dumps(selected_files[i]) + "\n")
        if i >= len(selected_files) * 0.6 and i < len(selected_files) * 0.8:
            with open(
                dataset + "/valid.jsonl",
                "a",
            ) as f:
                f.write(json.dumps(selected_files[i]) + "\n")
        if i >= len(selected_files) * 0.8:
            with open(
                dataset + "/test.jsonl",
                "a",
            ) as f:
                f.write(json.dumps(selected_files[i]) + "\n")

    # print("data preprocess finish")


def process_folder(folder_path):
    """
    Processes a folder to collect all unique CWE (Common Weakness Enumeration) prefixes from .txt files.

    Parameters
    ----------
    folder_path : str
        The path to the folder containing the text files.

    Returns
    -------
    None
    """
    all_part = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if os.path.isdir(file_path):
            process_folder(file_path)
        elif file_name.endswith(".txt"):

            with open(file_path, "r") as file:
                content = file.read()


            lines = content.splitlines()
            cwe_strings = [line for line in lines if line.startswith("CWE")]
            try:
                parts = cwe_strings[0].split("_")[0]
            except:
                continue
            all_part.append(parts)
    all_part = list(set(all_part))





def process_folder(folder_path, count_dict):
    """
    Counts occurrences of each CWE prefix in .txt files within a folder and its subfolders,
    storing counts in the provided dictionary.

    Parameters
    ----------
    folder_path : str
        The path to the folder containing the text files.
    count_dict : dict
        Dictionary to store the count of each CWE prefix.

    Returns
    -------
    None
    """
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if os.path.isdir(file_path):
            process_folder(file_path, count_dict)
        elif file_name.endswith(".txt"):

            with open(file_path, "r") as file:
                content = file.read()
            lines = content.splitlines()
            cwe_strings = [line for line in lines if line.startswith("CWE")]
            if cwe_strings:
                parts = cwe_strings[0].split("_")[0]
                count_dict[parts] = count_dict.get(parts, 0) + 1





def copy_files_with_prefix(source_folder, destination_folder, prefix):
    """
    Copies files containing a specific prefix within their content from the source folder to the destination folder.

    Parameters
    ----------
    source_folder : str
        The path to the source folder containing files.
    destination_folder : str
        The path to the destination folder where files will be copied.
    prefix : str
        The prefix to search for within file contents.

    Returns
    -------
    None
    """
    for root, dirs, files in os.walk(source_folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            with open(file_path, "r") as file:
                content = file.read()

            if prefix in content:
                destination_file = os.path.join(destination_folder, file_name)
                shutil.copy2(file_path, destination_file)

