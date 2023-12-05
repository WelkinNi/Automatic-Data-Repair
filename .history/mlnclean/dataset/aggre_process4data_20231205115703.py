import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import shutil
import copy
import os
import csv
import re

def add_id(input_file, output_file):
    with open(input_file, "r") as f_in, open(output_file, "w", newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

    # Write the header row with the new "id" field
        header = next(reader)
        header.insert(0, "ID")
        writer.writerow(header)

    # Write each row with the row number as the "id" field
        for i, row in enumerate(reader, start=1):
            row.insert(0, i)
            writer.writerow(row)


def split_data(base_path, name):
    # read the CSV file into a Pandas dataframe
    file_path = os.path.join(base_path, name) + "/" + str(name) + ".csv"
    data = pd.read_csv(file_path)
    data.insert(0, 'ID', range(1, len(data)+1))
    data_backup = copy.deepcopy(data)

    # shuffle the dataframe rows randomly
    data = data.sample(frac=1, random_state=42)

    # set the proportion of data to use for testing
    test_proportion = 1-float(20/len(data))

    # split the data into test and train sets
    test_data = data[:int(len(data)*test_proportion)]
    train_data = data[int(len(data)*test_proportion):]

    # write the test and train data to CSV files
    data_backup.to_csv(os.path.join(base_path, name) + "/" + "testData.csv", index=False)
    train_data.to_csv(os.path.join(base_path, name) + "/" + "trainData.csv", index=False)


def transfer_rules(base_path):
    input_path = base_path + "dc_rules_holoclean.txt"
    with open(input_path, 'r') as f:
        out_str_list_first = []
        out_str_list_fd = []
        for line in f.readlines():
            out_str_list_first.append(transfer_rule_first(line))
            out_str_list_fd.append(transfer_rule(line))
        f.close()
    output_path = base_path + "rules-first-order.txt"
    with open(output_path, 'w') as f:
        for line in out_str_list_first:
            f.write(line)
        f.close()
    output_path = base_path + "rules.txt"
    with open(output_path, 'w') as f:
        for line in out_str_list_fd:
            f.write(line)
        f.close()


def transfer_rule(input_str):
    output_str_left = ''
    output_str_right = ''
    # Extract the table names and join condition from the input string using regular expressions
    table_names = re.findall(r't\d+', input_str)
    join_cond = re.findall(r'(EQ|IQ)\((.*?)\)', input_str)
    # Generate the output string by iterating over the join conditions
    for cond_type, cond in join_cond:
        cond_parts = cond.split(',')
        col1 = re.sub(r't\d+\.', '', cond_parts[0])
        col2 = re.sub(r't\d+\.', '', cond_parts[1])
        if cond_type == 'EQ':
            output_str_left += f'{col1}(value{col1}),'
        elif cond_type == 'IQ':
            output_str_right += f'{col1}(value{col1})\n'
        output_str = output_str_left[:-1] + ' => ' + output_str_right
    return output_str


def transfer_rule_first(input_str):
    output_str = ''
    # Extract the table names and join condition from the input string using regular expressions
    table_names = re.findall(r't\d+', input_str)
    join_cond = re.findall(r'(EQ|IQ)\((.*?)\)', input_str)

    # Generate the output string by iterating over the join conditions
    for idx, j_cond in enumerate(join_cond):
        cond_type, cond = j_cond[0], j_cond[1]
        cond_parts = cond.split(',')
        col1 = re.sub(r't\d+\.', '', cond_parts[0])
        col2 = re.sub(r't\d+\.', '', cond_parts[1])
        if cond_type == 'EQ':
            output_str += f'!{col1}(value{col1})'
        elif cond_type == 'IQ':
            output_str += f'{col1}(value{col1})'
        if idx == len(join_cond) - 1:
            output_str += '\n'
            break
        else:
            output_str += ' v '
    return output_str

def copy_files_to_subfolders(source_folder):
    # Iterate over all the files in the source folder
    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)
        
        # Check if the file is not a subdirectory
        if os.path.isfile(source_file):
            # Iterate over all the subfolders within the source folder
            for subfolder_name in os.listdir(source_folder):
                subfolder_path = os.path.join(source_folder, subfolder_name)
                
                # Check if the item in the source folder is a subfolder
                if os.path.isdir(subfolder_path):
                    # Copy the file to the subfolder
                    shutil.copy(source_file, subfolder_path)


if __name__ == '__main__':
    name = "tax"
    base_path = f"/data/nw/DC_ED/References_inner_and_outer/mlnclean/dataset/{name}/dataset/"

    # src_dirty = f"./data with dc_rules/{name}/dirty.csv" 
    # tar_dirty = f"/data/nw/DC_ED/References_inner_and_outer/mlnclean/dataset/{name}/dataset/dirty.csv"
    src_clean = f"./data with dc_rules/{name}/clean.csv" 
    tar_clean = f"/data/nw/DC_ED/References_inner_and_outer/mlnclean/dataset/{name}/dataset/clean.csv"
    src_dc = f"./data with dc_rules/{name}/dc_rules_holoclean.txt" 
    tar_dc = f"/data/nw/DC_ED/References_inner_and_outer/mlnclean/dataset/{name}/dataset/dc_rules_holoclean.txt"
    tar_folder = f"/data/nw/DC_ED/References_inner_and_outer/mlnclean/dataset/{name}/noise"
    if name == "tax":
        src_folder = f"./data with dc_rules/{name}/runtime_noise" 
    else:
        src_folder = f"./data with dc_rules/{name}/noise" 
        
    # Copy Noise Files
    shutil.copytree(src_folder, tar_folder)
    
    # Move Data and Change Folder name
    source_dir = f"/data/nw/DC_ED/References_inner_and_outer/mlnclean/dataset/{name}"
    noise_dir = os.path.join(source_dir, "noise")
    dataset_dir = os.path.join(source_dir, "dataset")
    for file in os.listdir(noise_dir):
        if file.startswith("raha") and os.path.isdir(os.path.join(noise_dir, file)):
            shutil.rmtree(os.path.join(noise_dir, file))
        if file.endswith(".csv"):
            folder = os.path.join(noise_dir, os.path.splitext(file)[0])
            os.makedirs(folder, exist_ok=True)
            shutil.move(os.path.join(noise_dir, file), folder)
    os.rename(noise_dir, dataset_dir)
    
    # Copy Files
    shutil.copy(src_clean, tar_clean)
    shutil.copy(src_dc, tar_dc)

    # Generate Query DB
    data = pd.read_csv(base_path + "clean.csv")
    with open(base_path+"query.db", 'w') as f:
        for col in data.columns:
            str_col = col + f'(value{col})\n'
            f.write(str_col)
        f.close
    
    # Add ID to data
    input_file = base_path + 'clean.csv'
    output_file = base_path + 'ground_truth-hasID.csv'
    add_id(input_file, output_file)

    # Split Data
    for subfolder_name in os.listdir(base_path):
        sub_path = os.path.join(base_path, subfolder_name)
        # Check if the item in the source folder is a subfolder
        if os.path.isdir(sub_path):
            for filename in os.listdir(sub_path):
                if filename.endswith(".csv"):
                    split_data(base_path, filename.split('.')[0])

    # Transfer Rules
    transfer_rules(base_path)

    # Copy files to subfolders
    copy_files_to_subfolders(base_path)