import os
import json

trans_error = 'outer_error'
def file_transfer(input_dir):
    # Loop through all JSON files in input dir
    for filename in os.listdir(input_dir):
        if filename.endswith('.json') and filename.startwith('tax'):
            # Load JSON file
            with open(os.path.join(input_dir, filename)) as f:
                data = json.load(f)
                
            # Update CSV file name in source.file
            csv_file = data['source']['file'][0]
            updated_csv = csv_file.replace('inner_outer_error', trans_error)
            data['source']['file'][0] = updated_csv
            
            # Write out new JSON file
            new_filename = filename.replace('inner_outer_error', trans_error)
            with open(os.path.join(input_dir, new_filename), 'w') as f:
                json.dump(data, f, indent=4)
            
if __name__ == '__main__':
    input_dir = './NADEEF-master/EXP_json'
    file_transfer(input_dir)