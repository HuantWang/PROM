import os
import subprocess

# Define the target directory and file
target_directory = os.path.abspath("../case_study/DeviceM/")
target_file = "ae_dev.sh"

# Check if the target file exists, if not, change to the target directory
if not os.path.isfile(target_file):
    os.chdir(target_directory)
    print(f"Switched to directory: {target_directory}")

# Run the shell script using subprocess
subprocess.run(["bash", "ae_dev.sh"])

# Change back to the original directory if the target file exists
if os.path.isfile(target_file):
    os.chdir("../../tutorial")
