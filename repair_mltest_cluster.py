def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import argparse
import sys
import warnings
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans,SpectralClustering,AffinityPropagation
from sklearn.metrics import silhouette_score
from rich.progress import track
from tqdm import tqdm
import logging
logging.getLogger().setLevel(logging.ERROR)

def kmeans(X, y, num_classes):
    estimator = KMeans(n_clusters=num_classes)
    estimator.fit(X)
    labels = estimator.labels_
    if len(set(labels)) > 1:
        silhouette = silhouette_score(X, labels)
    else:
        silhouette = -1
    return silhouette

def spectral(X, y, num_classes):
    estimator = SpectralClustering(n_clusters=num_classes)
    estimator.fit(X)
    labels = estimator.labels_
    if len(set(labels)) > 1:
        silhouette = silhouette_score(X, labels)
    else:
        silhouette = -1
    return silhouette

def affinityprop(X, y, num_classes):
    estimator = AffinityPropagation(preference=-50, random_state=0)
    estimator.fit(X)
    labels = estimator.labels_
    if len(set(labels)) > 1:
        silhouette = silhouette_score(X, labels)
    else:
        silhouette = -1
    return silhouette

def testing_func(rep_df, clean_df, target, feature_schema):
    feature_schema = [x.lower() for x in feature_schema]
    rep_df.columns = rep_df.columns.str.lower()
    clean_df.columns = clean_df.columns.str.lower()
    target = target.lower()
    df_encoded = pd.get_dummies(rep_df[feature_schema])
    sanitized_feature_names = {}
    for feature_names_str in df_encoded.columns:
        valid_chars = [char for char in feature_names_str if char not in ['[', ']', '<']]
        sanitized_feature_names[feature_names_str] = ''.join(valid_chars)

    df_encoded.rename(columns=sanitized_feature_names, inplace=True)
    X = df_encoded
    y = rep_df[target]
    num_classes = len(y.unique())
    res_dict = {}
    silhouette = kmeans(X, y, num_classes)
    res_dict['kmeans'] = [silhouette]
    silhouette = spectral(X, y, num_classes)
    res_dict['spectral'] = [silhouette]
    silhouette = affinityprop(X, y, num_classes)
    res_dict['affinityprop'] = [silhouette]
    return res_dict


if __name__ == "__main__":
    # dirty_path = "./DATASET/data with dc_rules/flights/noise/flights-inner_error-20.csv"
    # clean_path = "./DATASET/data with dc_rules/flights/clean.csv"
    # rep_path = "./DATASET/Repaired_res/bigdansing/flights/repaired_flights1-inner_error-10.csv"

    target_dict = {'Adult':'Income', 'Dress':'Recommendation'}
    # target_dict = {'flights':'flight', 'hospital':'HospitalName', 'beers':'style', 'rayyan':'article_language'}
    # target_dict = {'flights':'flight'}

    data_base_dir = "./DATASET/data with dc_rules/"
    rep_base_dir = "./DATASET/Repaired_res/"
    res_base_dir = "./DATASET/Exp_result_downstream/"
    datasets = target_dict.keys()
    error_types = ["-inner_outer_error-",]
    
    methods = ['bigdansing', 'holistic', 'horizon', 'nadeef', 'mlnclean', 'scared', 'raha_baran', 'boostclean', 'Unified','holoclean', ]
    
    for k in range(1,4):
        for data in datasets:
            clean_path = data_base_dir + data + "/" + 'clean' + ".csv"
            clean_df = pd.read_csv(clean_path).astype(str)
            clean_df.fillna('nan', inplace=True)
            target = target_dict[data]

            feature_schema = list(clean_df.columns)
            feature_schema.remove(target)

            # # using clean data
            # res_dict = testing_func(clean_df, clean_df, target, feature_schema)
            # for algm in res_dict:
            #     res = res_base_dir + "clean_dirty/" + data + '/cluster-' + algm + data + str(k) + 'clean' + ".csv"
            #     f = open(res, 'w')
            #     sys.stdout = f
            #     print("{pre}".format(pre=res_dict[algm][0]))
            #     f.close()
            #     sys.stdout = open("/dev/stdout", "w")

            for err_type in error_types:
                error_rate = ['01']
                error_rate = [10, 30, 50, 70, 90]
                for err_r in error_rate:
                    dirty_path = data_base_dir + data + "/" + 'noise' + "/" + data + err_type + str(err_r) + ".csv"
                    dirty_df = pd.read_csv(dirty_path).astype(str)
                    dirty_df.fillna('nan', inplace=True)
                    
                    # using dirty data
                    res_dict = testing_func(dirty_df, clean_df, target, feature_schema)
                    for algm in res_dict:
                        res = res_base_dir + "clean_dirty/" + data + '/' + "cluster-" + algm + data + str(k) + err_type + str(err_r) + ".csv"
                        f = open(res, 'w')
                        sys.stdout = f
                        print("{pre}".format(pre=res_dict[algm][0]))
                        f.close()
                        sys.stdout = open("/dev/stdout", "w")

                    # # using rep data
                    # print("\n*************The %d th Experiment: Dataset: %s || Error Type: %s || Error Rate: %s *************" % (k, data, err_type, err_r))
                    # for method in tqdm(methods, ncols=80):
                    #     print('\n')
                    #     rep_path = rep_base_dir + method + "/" + data + "/" + 'repaired' + '_' + data + str(k) + err_type + str(err_r) + ".csv"
                    #     # rep_path = rep_base_dir + method + "/" + data + "/" + 'Raha_improved-' + data + str(k) + err_type + str(err_r) + ".csv"
                    #     try:
                    #         flag = 0
                    #         rep_df = pd.read_csv(rep_path).astype(str)
                    #         rep_df.fillna('nan', inplace=True)
                    #     except Exception as e:
                    #         print(rep_path + " 【does not exist】")
                    #         flag = 1
                    #     if flag == 0:
                    #         try:
                    #             res_dict = testing_func(rep_df, clean_df, target, feature_schema)
                    #             for algm in res_dict:
                    #                 res = res_base_dir + method + "/" + data + '/cluster-' + algm + '_repaired_' + data + str(k) + err_type + str(err_r) + ".csv"
                    #                 # res = res_base_dir + method + "/" + data + '/cluster-' + algm + 'raha_improved-' + data + str(k) + err_type + str(err_r) + ".csv"
                    #                 f = open(res, 'w')
                    #                 sys.stdout = f
                    #                 print("{pre}".format(pre=res_dict[algm][0]))
                    #                 f.close()
                    #                 sys.stdout = open("/dev/stdout", "w")
                    #         except Exception as e:
                    #             logging.error("【An error occurred】: %s %s", e, rep_path)