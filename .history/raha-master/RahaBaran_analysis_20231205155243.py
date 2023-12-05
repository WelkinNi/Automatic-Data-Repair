
# %%
from json.tool import main
from transformers import BertTokenizer, AutoTokenizer
import line_profiler as lp
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import fasttext
import math
import time
import random
import copy
from tqdm import tqdm
from transformers import AutoTokenizer
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
import pandas
import IPython.display
import raha
# %%

def FT_emd(col, data):
    vecs_col = None
    for i in range(len(data)):
        vec = None
        str_i = str(data.loc[i, col]).strip().split()
        for s in str_i:
            if s in wv_from_text.key_to_index.keys():
                s_v = wv_from_text[s]
            else:
                s_v = wv_from_text['nan']
            if vec is None:
                vec = s_v
            else:
                vec = vec + s_v
        if len(str_i) > 0:
            vec = vec / len(str_i)
        else:
            vec = wv_from_text['nan']
        if vecs_col is None:
            vecs_col = vec
            vecs_col = vecs_col.reshape((1,-1))
        else:
            vecs_col = np.concatenate((vecs_col, vec.reshape((1,-1))), axis=0)
    return vecs_col

app_1 = raha.Detection()

app_1.LABELING_BUDGET = 30

app_1.VERBOSE = True
clean_filepath = "./raha-master/datasets/hospital/clean.csv"
dirty_filepath = "./raha-master/datasets/hospital/dirty.csv"



dataset_dictionary = {
    "name": "rayyan",
    "path": dirty_filepath,
    "clean_path": clean_filepath
}
d = app_1.initialize_dataset(dataset_dictionary)

app_1.run_strategies(d)
app_1.generate_features(d)

# d.columns_features_list get the feature vectors

app_1.build_clusters(d)

# %%
while len(d.labeled_tuples) < app_1.LABELING_BUDGET:
    app_1.sample_tuple(d)
    if d.has_ground_truth:
        app_1.label_with_ground_truth(d)
    else:
        print("Label the dirty cells in the following sampled tuple.")
        sampled_tuple = pandas.DataFrame(data=[d.dataframe.iloc[d.sampled_tuple, :]], columns=d.dataframe.columns)
        IPython.display.display(sampled_tuple)
        for j in range(d.dataframe.shape[1]):
            cell = (d.sampled_tuple, j)
            value = d.dataframe.iloc[cell]
            correction = input("What is the correction for value '{}'? Type in the same value if it is not erronous.\n".format(value))
            user_label = 1 if value != correction else 0
            d.labeled_cells[cell] = [user_label, correction]
        d.labeled_tuples[d.sampled_tuple] = 1

# %%
app_1.propagate_labels(d)

# app_1.predict_labels(d)
detected_cells_dictionary = {}

con_df = pd.DataFrame(columns = d.dataframe.columns)
for j in range(d.dataframe.shape[1]):
    feature_vectors = d.column_features[j]
    conf = np.zeros((d.dataframe.shape[0], 1))
    x_train = [feature_vectors[i, :] for i in range(d.dataframe.shape[0]) if (i, j) in d.extended_labeled_cells]
    y_train = [d.extended_labeled_cells[(i, j)] for i in range(d.dataframe.shape[0]) if
               (i, j) in d.extended_labeled_cells]
    x_test = feature_vectors
    if len(pd.DataFrame(y_train).value_counts()) == 1:
        conf = 1-np.zeros((d.dataframe.shape[0], 1))
        con_df[d.dataframe.columns[j]] = pd.DataFrame(conf)
        continue
    clf = LogisticRegression(random_state=100, max_iter=3).fit(x_train, y_train)
    predicted_labels = clf.predict(feature_vectors)
    predict_prob = clf.predict_proba(feature_vectors)[:,0].reshape((-1, 1))
    con_df[d.dataframe.columns[j]] = pd.DataFrame(predict_prob)
    for i, pl in enumerate(predicted_labels):
        if (i in d.labeled_tuples and d.extended_labeled_cells[(i, j)]) or (i not in d.labeled_tuples and pl):
            detected_cells_dictionary[(i, j)] = "JUST A DUMMY VALUE"
d.detected_cells.update(detected_cells_dictionary)

p, r, f = d.get_data_cleaning_evaluation(d.detected_cells)[:3]
print("Raha's performance on {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(d.name, p, r, f))

# %%
"""
Using Fasttext to improve
"""
# print("-"*40)
# dimension = 300
# wv_from_text = KeyedVectors.load_word2vec_format('./wiki-news-300d-1M.vec', binary=False,limit=1000000)
# model = fasttext.train_unsupervised(dirty_filepath, model='skipgram',dim=dimension,lr=0.0001, epoch=10, ws=10)
# ftvecs = None
# for col in d.dataframe.columns:
#     if ftvecs is None:
#         ftvecs = FT_emd(col, d.dataframe).reshape((1,-1,dimension))
#     else:
#         ftvecs = np.concatenate((ftvecs, FT_emd(col, d.dataframe).reshape((1,-1,dimension))), axis=0)

# for i in range(len(d.column_features)):
#     d.column_features[i] = np.concatenate((d.column_features[i], ftvecs[i]), axis=1)
# # d.columns_features_list get the feature vectors

# for i in range(len(d.column_features)):
#     d.column_features[i] = np.concatenate((d.column_features[i], ftvecs[i]), axis=1)

# app_1.build_clusters(d)

# # %%
# while len(d.labeled_tuples) < app_1.LABELING_BUDGET:
#     app_1.sample_tuple(d)
#     if d.has_ground_truth:
#         app_1.label_with_ground_truth(d)
#     else:
#         print("Label the dirty cells in the following sampled tuple.")
#         sampled_tuple = pandas.DataFrame(data=[d.dataframe.iloc[d.sampled_tuple, :]], columns=d.dataframe.columns)
#         IPython.display.display(sampled_tuple)
#         for j in range(d.dataframe.shape[1]):
#             cell = (d.sampled_tuple, j)
#             value = d.dataframe.iloc[cell]
#             correction = input("What is the correction for value '{}'? Type in the same value if it is not erronous.\n".format(value))
#             user_label = 1 if value != correction else 0
#             d.labeled_cells[cell] = [user_label, correction]
#         d.labeled_tuples[d.sampled_tuple] = 1

# # %%
# app_1.propagate_labels(d)

# app_1.predict_labels(d)

# # %%
# p, r, f = d.get_data_cleaning_evaluation(d.detected_cells)[:3]
# print("Raha's performance on {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(d.name, p, r, f))

# raha_label_cells = list(set([t[0] for t in list(d.labeled_cells.keys())]))
# actual_errors = d.get_actual_errors_dictionary()

# cols = list(d.dataframe.columns)
# col_eva = {}
# for c in cols:
#     idx = cols.index(c)
#     output_size = 0
#     ac_errors = 0
#     ec_tp = 0
#     for cell in actual_errors:
#         if cell[1] == idx:
#             ac_errors = ac_errors + 1
#     for cell in d.detected_cells:
#         if cell[1] == idx:
#             output_size = output_size + 1
#             if cell in actual_errors:
#                 ec_tp = ec_tp + 1
#     col_eva[c] = {}
#     if output_size == 0:
#         col_eva[c]['prec'] = 0
#     else:
#         col_eva[c]['prec'] = ec_tp / output_size
#     col_eva[c]['rec'] = ec_tp / ac_errors
#     col_eva[c]['F1'] = 2*col_eva[c]['prec']*col_eva[c]['rec']/(col_eva[c]['rec']+col_eva[c]['prec']+1e-12)

# print("-"*30 + "Error cells missing:" + "-"*30)
# for cell in actual_errors:
#     if cell not in d.detected_cells:
#         print(cell)

# print("-"*30 + "Cells should be right:" + "-"*30)
# for cell in d.detected_cells:
#     if cell not in actual_errors:
#         print(cell)

print("Done")