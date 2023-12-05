import os
import csv
import shutil


def process_file(file_path, new_file_name):
    with open(file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        lines = list(reader)
    # Remove the first column
    cleaned_lines = [line[1:] for line in lines]
    # Write the cleaned lines to the new file
    with open(new_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(cleaned_lines)
    

if __name__ == "__main__":
    task_name = "rayyan"
    base_dir = f"/data/nw/DC_ED/References_inner_and_outer/mlnclean/dataset/{task_name}/dataset"
    tar_dir = f"./Repaired_res/mlnclean/{task_name}"

    for root, subdirs, filenames in os.walk(base_dir):
        for subdir in subdirs:
            for _, _, subfiles in os.walk(os.path.join(root, subdir)):
                for subfile in subfiles:
                    if subfile == "RDBSCleaner_cleaned.txt":
                        rep_file_path = os.path.join(os.path.join(root, subdir), subfile)
                        dir_name = os.path.basename(subdir)
                        if task_name != "tax":
                            new_file_name = "repaired_" + dir_name.split('-')[0] + "2-" + dir_name.split('-')[1] + "-" + dir_name.split('-')[2] + ".csv"
                        else:
                            new_file_name = "repaired_" + dir_name.split('-')[0] + "3-" + dir_name.split('-')[2] + "-" + dir_name.split('-')[3] + ".csv"
                        new_file_path = os.path.join(tar_dir, new_file_name)
                        process_file(rep_file_path, new_file_path)
                        
