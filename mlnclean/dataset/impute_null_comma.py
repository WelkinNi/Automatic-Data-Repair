import csv
import os

def replace_comma_and_null_with_string(file_path):
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        modified_rows = []
        for row in reader:
            replaced_nan_row = ['nan' if cell is None or cell == '' else cell for cell in row]
            m_row = []
            for item in replaced_nan_row:
                item = item.replace(',', '')
                item = item.replace('"', '')
                m_row.append(item)
            modified_rows.append(m_row)

    # Write the updated data back to the CSV file
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(modified_rows)

def process_csv_files(folder_path):
    # Process each file in the folder and its subfolders
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                replace_comma_and_null_with_string(file_path)
                # print(f'Replaced null values in: {file_path}')
        for dir in dirs:
            for _, _, subfiles in os.walk(os.path.join(root, dir)):
                for subfile in subfiles:
                    if subfile.endswith('.csv'):
                        file_path = os.path.join(os.path.join(root, dir), file)
                        replace_comma_and_null_with_string(file_path)


if __name__ == '__main__':
    # Specify the folder path where the CSV files are located
    name = "flights"
    base_path = f"/data/nw/DC_ED/References_inner_and_outer/mlnclean/dataset/{name}/dataset/"
    # Call the function to process the CSV files
    process_csv_files(base_path)
