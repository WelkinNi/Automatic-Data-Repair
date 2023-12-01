
# import pandas as pd
# clean_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/data with dc_rules/tax/clean.csv"
# clean_df = pd.read_csv(clean_path)

rule_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/data with dc_rules/hospital/dc_rules-validate-fd-horizon.txt"
with open(rule_path, 'r') as input_file:
    lines = input_file.readlines()
input_file.close()

for i in range(len(lines)):
    out_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/data with dc_rules/hospital/multi_rules/dc_rules-validate-fd-horizon-" + str(i+1) + ".txt"
    with open(out_path, 'w') as output_file:
        for line in lines[:i+1]:
            output_file.write(line)
    output_file.close()