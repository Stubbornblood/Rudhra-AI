import os
import threading
import psutil
from pathlib import Path
def find_file(file_name, drive_choice="all"):
    partitions = psutil.disk_partitions()

    if drive_choice.lower() == "all":
        drives_to_search = [Path(partition.mountpoint) for partition in partitions]
    else:
        valid_drives = [p.device[0].upper() for p in partitions]
        if drive_choice.upper() not in valid_drives:
            if drive_choice.upper()[0] not in valid_drives:
                print(f"'{drive_choice}' is not a valid drive letter.")
                return
            drives_to_search = [Path(drive_choice.upper()[0] + ":")]
        else:
            drives_to_search = [Path(drive_choice.upper() + ":")]

    results = []
    for drive_path in drives_to_search:
        for root, dirs, files in os.walk(drive_path):
            for file in files:
                if file == file_name:
                    results.append(os.path.abspath(os.path.join(root, file)))

    if len(results) == 0:
        print(f"No files found with the name '{file_name}'.")
        return

    if len(results) == 1:
        return results[0]

    print(f"Found {len(results)} files with the name '{file_name}'.")
    for i, result in enumerate(results):
        print(f"{i+1}. {result}")

    parent_folder_name = input("Enter the name of the parent folder (case-sensitive): ")
    filtered_results = [result for result in results if os.path.basename(os.path.dirname(result)) == parent_folder_name]
    
    if len(filtered_results) == 0:
        print(f"No files found with the name '{file_name}' in a folder with the name '{parent_folder_name}'.")
        return

    if len(filtered_results) == 1:
        return filtered_results[0]

    print(f"Found {len(filtered_results)} files with the name '{file_name}' in a folder with the name '{parent_folder_name}'.")
    for i, result in enumerate(filtered_results):
        print(f"{i+1}. {result}")

    choice = input("Enter the number of the file you want to choose: ")
    try:
        choice_index = int(choice) - 1
        return filtered_results[choice_index]
    except:
        print("Invalid choice.")
        return
