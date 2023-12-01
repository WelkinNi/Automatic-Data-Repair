import os
import csv
import json
import pandas as pd

def change_file_name(directory, pattern):
    for filename in os.listdir(directory):
        if filename.startswith(pattern):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r') as f:
                    reader = csv.reader(f)
                    row = next(reader)  # Extract the first row
                    second = row[1].replace('"', '').lower()  # Extract the second field and remove double quotes
                    third = row[2].replace('"', '').lower()  # Extract the third field and remove double quotes
            except:
                second = 'none'
                third = 'none'
            new_filename = f'violation_nadeef_{second}_{third}.csv'
            new_filepath = os.path.join(directory, new_filename)
            os.rename(filepath, new_filepath)
            f.close()

def find_violations(filepath):
    violations = {}
    data_df = pd.read_csv(filepath, header=None)
    for i in range(len(data_df)):
        if data_df.iloc[i, 0] not in violations.keys():
            violations[data_df.iloc[i, 0]] = set()
        else:
            violations[data_df.iloc[i, 0]].add(data_df.iloc[i, 3])
    return violations


if __name__ == '__main__':
    directory = '/data/nw/DC_ED/References_inner_and_outer/NADEEF-master/out'  # Replace with the path to the directory you want to search
    json_dir = '/data/nw/DC_ED/References_inner_and_outer/NADEEF-master/EXP_json'
    pattern = 'violation_nadeef_'
    dataname_list = ['beers', 'flights', 'hos', 'rayyan']
    dataset_list = ['beers', 'flights', 'hospital', 'rayyan']
    data_file = {
        'beers': '/data/nw/DC_ED/References_inner_and_outer/NADEEF-master/EXP_json/data with dc_rules/beers/clean.csv',
        'flights': '/data/nw/DC_ED/References_inner_and_outer/NADEEF-master/EXP_json/data with dc_rules/flights/clean.csv',
        'hospital': '/data/nw/DC_ED/References_inner_and_outer/NADEEF-master/EXP_json/data with dc_rules/hospital/clean.csv',
        'rayyan': '/data/nw/DC_ED/References_inner_and_outer/NADEEF-master/EXP_json/data with dc_rules/rayyan/clean.csv'
    }
    json_file = {
        'beers': '/data/nw/DC_ED/References_inner_and_outer/NADEEF-master/EXP_json/beers_inner_outer_error_01.json',
        'flights': '/data/nw/DC_ED/References_inner_and_outer/NADEEF-master/EXP_json/flights_inner_outer_error_01.json',
        'hospital': '/data/nw/DC_ED/References_inner_and_outer/NADEEF-master/EXP_json/hospital_inner_outer_error_01.json',
        'rayyan': '/data/nw/DC_ED/References_inner_and_outer/NADEEF-master/EXP_json/rayyan_inner_outer_error_01.json'
    }

    # get cols for each dataset
    data_cols = {}
    for key, val in data_file.items():
        cols = list(pd.read_csv(val).columns)
        data_cols[key] = cols
    
    # change file name into readable ones
    for d in dataname_list:
        pattern_d = pattern + "d"
        change_file_name(directory, pattern)
    
    # read json file
    d_rule_dict = {}
    for d in dataset_list:
        json_path = json_file[d]
        with open(json_path, 'r') as f:
            d_json = json.load(f)
        f.close()
        d_rule_dict[d] = d_json

    # generate final violations
    dirty_vios = {}
    for d in dataset_list:
        for err in ['01', '10', '30', '50', '70', '90']:
            # for err in ['90']:
            vio_list = set()
            dirty_data = d + '_inner_outer_error_' + err + '.csv'
            for i in range(len(d_rule_dict[d]["rule"])):
                # for i in range(1):
                vio_data = "violation_nadeef_" + d_rule_dict[d]["rule"][i]['name'].lower()
                vio_data = vio_data + '_tb_' + d + '_inner_outer_error_' + err + '.csv'
                vio_path = os.path.join(directory, vio_data)
                try:
                    violations = find_violations(vio_path)
                    result_attr = d_rule_dict[d]["rule"][i]['value'][0].split('.')[-1][:-1]
                    result_attr_idx = data_cols[d].index(result_attr)
                    for key, val in violations.items():
                        val = list(val)
                        min_v = min(val)
                        max_v = max(val)
                        vio_list.add((min_v, max_v, result_attr_idx))
                except:
                    print(vio_path + 'does not exists!')
            dirty_vios[dirty_data] = vio_list

    # print final violations to file
    for key, val in dirty_vios.items():
        with open(os.path.join(json_dir, key[:-4]+'.txt'), 'w') as f:
            key_str = str(key)
            val_str = '\n'.join(str(x[0])+','+str(x[1])+','+str(x[2]) for x in val)
            f.write(val_str + '\n')
        f.close()