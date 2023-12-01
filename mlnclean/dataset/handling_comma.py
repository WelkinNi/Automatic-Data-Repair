import csv
import os

def replace_comma(file_path):
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        modified_rows = []
        for row in csv_reader:
            m_row = []
            for item in row:
                
                # Check if the item is enclosed in double quotes
                
                # Remove commas within the item
                item = item.replace(',', '')
                item = item.replace('"', '')
                
                m_row.append(item)
            modified_rows.append(m_row)

    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(modified_rows)

if __name__ == '__main__':
    # Specify the folder path where the CSV files are located

    # name = "rayyan"
    # base_path = f"/data/nw/DC_ED/References_inner_and_outer/mlnclean/dataset/{name}/dataset/"

    # # Call the function to process the CSV files
    # process_csv_files(base_path)

    name = "hospital"

    file_path = f"/data/nw/DC_ED/References_inner_and_outer/mlnclean/dataset/{name}/dataset/{name}-inner_outer_error-01/testData.csv"
    replace_comma(file_path)

    file_path = f"/data/nw/DC_ED/References_inner_and_outer/mlnclean/dataset/{name}/dataset/{name}-inner_outer_error-01/trainData.csv"
    replace_comma(file_path)
    
    file_path = f"/data/nw/DC_ED/References_inner_and_outer/mlnclean/dataset/{name}/dataset/{name}-inner_outer_error-01/clean.csv"
    replace_comma(file_path)