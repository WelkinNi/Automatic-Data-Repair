#!/usr/bin/env python
from activedetect.loaders.csv_loader import CSVLoader
from sklearn.ensemble import RandomForestClassifier
from activedetect.experiments.Experiment import Experiment
from activedetect.loaders.type_inference import LoLTypeInference
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

"""
Example Experiment Script
"""
def getDataType(data):
    data_type = LoLTypeInference().getDataTypes(data.values)
    cat_type_col = []
    num_type_col = []
    string_type_col = []
    col2type = {}
    for i in range(len(data.columns)):
        col2type[data.columns[i]] = data_type[i]
        if data_type[i] == 'categorical':
            cat_type_col.append(data.columns[i])
        elif data_type[i] == 'numerical':
            num_type_col.append(data.columns[i])
        elif data_type[i] == 'string':
            string_type_col.append(data.columns[i])
    return col2type, cat_type_col, num_type_col, string_type_col

def getCatMask(clean_df, dirty_df):
    dim = len(clean_df)
    mask_metric = np.zeros_like(clean_df.values)
    df_index = list(clean_df.index)
    df_columns = list(clean_df.columns)
    for i in range(len(df_index)):
        for j in range(len(df_columns)):
            if (clean_df.iloc[i, j] == dirty_df.iloc[i, j]):
                mask_metric[i, j] = 1
            else:
                mask_metric[i, j] = 0
    mask_metric = pd.DataFrame(mask_metric, columns=clean_df.columns)
    return mask_metric

def getNumMask(clean_df, dirty_df):
    dim = len(clean_df)
    mask_metric = np.zeros_like(clean_df.values)
    df_index = list(clean_df.index)
    df_columns = list(clean_df.columns)
    for i in range(len(df_index)):
        for j in range(len(df_columns)):
            try:
                if (float(clean_df.iloc[i, j]) - float(dirty_df.iloc[i, j]) < 0.01):
                    mask_metric[i, j] = 1
                else:
                    mask_metric[i, j] = 0
            except:
                mask_metric[i, j] = 0
    mask_metric = pd.DataFrame(mask_metric, columns=clean_df.columns)
    return mask_metric

def getCatTrainMask(real_mask, column_num_dict):
    total = []
    n_mask = len(real_mask)
    if len(column_num_dict) > 0:
        columns_name = real_mask.columns
        total = np.ones((n_mask, len(column_num_dict[columns_name[0]])))
        for i in range(len(real_mask)):
            if real_mask.iloc[i, 0] == 0:
                for j in range(len(total[i])):
                    total[i, j] = 0

        for num in range(1, len(columns_name)):
            temp = np.ones((n_mask, len(column_num_dict[columns_name[num]])))
            for i in range(1, n_mask):
                if real_mask.iloc[i, num] == 0:
                    for j in range(len(temp[i])):
                        temp[i, j] = 0
            total = np.concatenate((total, temp), axis=1)
    return total

def repairNan(data):
    impute_value = {}
    for col in data.columns:
        impute_value[col] = data[col].value_counts().index[0]
    data.fillna(value=impute_value, inplace=True)
    return data

def repairNumTypeError(data):
    columns = data.columns
    for col in columns:
        for i in range(len(data[col])):
            try:
                float(data.loc[i, col])
            except:
                repair = data[col].value_counts().index[0]
                data.loc[i, col] = repair
    return data

def getCatMasknp(clean_np, dirty_np):
    dim = len(clean_np)
    mask_metric = np.zeros_like(clean_np)
    for i in range(len(clean_np)):
        for j in range(len(clean_np[0])):
            if (clean_np[i, j] == dirty_np[i, j]):
                mask_metric[i, j] = 1
            else:
                mask_metric[i, j] = 0
    return mask_metric

if __name__ == "__main__":
    # Loads the first 100 lines of the dataset
    # loaded data is a list of lists [ [r1], [r2],...,[r100]]
    c = CSVLoader()
    loadedData = c.loadFile('/Users/mark/Downloads/GAN-Repair/GAN-Clean/datasets/Flights_dirty.csv')
    del loadedData[0]

    #all but the last column are features
    features = [l[0:-1] for l in loadedData]
    # features = np.array(features, dtype = float)
    # features = features.tolist()

    #last column is a label, turn into a float
    # labels = [l[-1] for l in loadedData]
    # labels = np.array(labels, dtype = int)
    # labels = labels.tolist()
    labels = [1.0*(l[-1].find("p.m.")==-1) for l in loadedData]

    # run the experiment, results are stored in uscensus.log
    # features, label, sklearn model, name
    e = Experiment(features, labels, RandomForestClassifier(), "uscensus")
    repair_features = e.runAllAccuracy()
    repair_total = np.concatenate((np.array(repair_features), np.array(loadedData)[:, -1].reshape(-1, 1)), axis=1)
    
    clean_filepath = '/Users/mark/Downloads/GAN-Repair/GAN-Clean/datasets/Flights_clean.csv'
    dirty_filepath = '/Users/mark/Downloads/GAN-Repair/GAN-Clean/datasets/Flights_dirty.csv'
    clean_df = pd.read_csv(clean_filepath)
    dirty_df = pd.read_csv(dirty_filepath)
    # print(dirty_df.info())

    # fill NAN
    clean_df = repairNan(clean_df)
    dirty_df = repairNan(dirty_df)
    
    # get data type
    cat_name = []
    num_name = []
    string_name = []
    col2type = {}
    col2type, cat_name, num_name, string_name = getDataType(clean_df)

    # get data val
    clean_cat_val = clean_df[cat_name]
    dirty_cat_val = dirty_df[cat_name]

    # Mask Metric
    mask_cat_real = getCatMask(clean_cat_val, dirty_cat_val)
    mask_truth_ori = mask_cat_real.values
    repair_mask = getCatMasknp(clean_df.values, repair_total)
    print("-----------------Repair Report-----------------")
    print(classification_report(mask_truth_ori.reshape(-1,).astype(int), repair_mask.reshape(-1,).astype(int)))

    print("Over")
