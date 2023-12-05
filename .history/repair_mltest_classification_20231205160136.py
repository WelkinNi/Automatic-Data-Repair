def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import argparse
import sys
import warnings
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neural_network import MLPRegressor, MLPClassifier
from rich.progress import track
from tqdm import tqdm
import logging
logging.getLogger().setLevel(logging.ERROR)

def xgbc(X_train, X_test, y_train, y_test):
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision, recall, f1 = evaluate(y_test, y_pred)
    return precision, recall, f1

def mlpc(X_train, X_test, y_train, y_test):
    model = MLPClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision, recall, f1 = evaluate(y_test, y_pred)
    return precision, recall, f1

def randomfc(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier(n_estimators=10)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test.to_numpy())
    precision, recall, f1 = evaluate(y_test, y_pred)
    return precision, recall, f1

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
    train_indices, test_indices = train_test_split(range(len(df_encoded)), test_size=0.2, random_state=0)
    X_train = df_encoded.iloc[train_indices]
    y_train = rep_df[target].iloc[train_indices]
    X_test = df_encoded.iloc[test_indices]
    y_test = clean_df[target].iloc[test_indices]
    res_dict = {}
    # pre, rec, f1 = randomfc(X_train, X_test, y_train, y_test)
    # res_dict['randomfc'] = [pre, rec, f1]
    pre, rec, f1 = mlpc(X_train, X_test, y_train, y_test)
    res_dict['mlpc'] = [pre, rec, f1]
    pre, rec, f1 = xgbc(X_train, X_test, y_train, y_test)
    res_dict['xgbc'] = [pre, rec, f1]
    return res_dict

def evaluate(y_test, y_pred):
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return precision,recall,f1


if __name__ == "__main__":
    target_dict = {'flights':'flight', 'hospital':'ProviderNumber', 'beers':'city', 'rayyan':'article_language'}

    data_base_dir = "./data_with_rules/"
    rep_base_dir = "./Repaired_res/"
    res_base_dir = "./Exp_result_downstream/"
    datasets = ['hospital', 'flights', 'beers', 'rayyan']
    error_types = ["-outer_error-", "-inner_error-", "-inner_outer_error-"]
    error_types = ["-inner_outer_error-",]
    
    methods = ['bigdansing', 'holistic', 'nadeef', 'mlnclean', 'horizon', 'raha_baran', 'scared', 'holoclean', 'Unified','boostclean']
    methods = ['daisy']
    
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
            #     res = res_base_dir + "clean_dirty/" + data + '/classi-' + algm + data + str(k) + 'clean' + ".csv"
            #     f = open(res, 'w')
            #     sys.stdout = f
            #     print("{pre}\n{rec}\n{f1}".format(pre=res_dict[algm][0], rec=res_dict[algm][1], f1=res_dict[algm][2]))
            #     f.close()
            #     sys.stdout = open("/dev/stdout", "w")

            for err_type in error_types:
                error_rate = [10, 30, 50, 70, 90]
                error_rate = ['01']
                for err_r in error_rate:
                    dirty_path = data_base_dir + data + "/" + 'noise' + "/" + data + err_type + str(err_r) + ".csv"
                    dirty_df = pd.read_csv(dirty_path).astype(str)
                    dirty_df.fillna('nan', inplace=True)
                    
                    # using dirty data
                    res_dict = testing_func(dirty_df, clean_df, target, feature_schema)
                    for algm in res_dict:
                        res = res_base_dir + "clean_dirty/" + data + '/' + "classi-" + algm + data + str(k) + err_type + str(err_r) + ".csv"
                        f = open(res, 'w')
                        sys.stdout = f
                        print("{pre}\n{rec}\n{f1}".format(pre=res_dict[algm][0], rec=res_dict[algm][1], f1=res_dict[algm][2]))
                        f.close()
                        sys.stdout = open("/dev/stdout", "w")

                    # using rep data
                    print("\n*************The %d th Experiment: Dataset: %s || Error Type: %s || Error Rate: %s *************" % (k, data, err_type, err_r))
                    for method in tqdm(methods, ncols=80):
                        print('\n')
                        rep_path = rep_base_dir + method + "/" + data + "/" + 'repaired' + '_' + data + str(k) + err_type + str(err_r) + ".csv"
                        # rep_path = rep_base_dir + method + "/" + data + "/" + 'Raha_improved-' + data + str(k) + err_type + str(err_r) + ".csv"
                        try:
                            flag = 0
                            rep_df = pd.read_csv(rep_path).astype(str)
                            rep_df.fillna('nan', inplace=True)
                        except Exception as e:
                            print(rep_path + " 【does not exist】")
                            flag = 1
                        if flag == 0:
                            try:
                                res_dict = testing_func(rep_df, clean_df, target, feature_schema)
                                for algm in res_dict:
                                    res = res_base_dir + method + "/" + data + '/classi-' + algm + '_repaired_' + data + str(k) + err_type + str(err_r) + ".csv"
                                    # res = res_base_dir + method + "/" + data + '/classi-' + algm + 'raha_improved-' + data + str(k) + err_type + str(err_r) + ".csv"
                                    f = open(res, 'w')
                                    sys.stdout = f
                                    print("{pre}\n{rec}\n{f1}".format(pre=res_dict[algm][0], rec=res_dict[algm][1], f1=res_dict[algm][2]))
                                    f.close()
                                    sys.stdout = open("/dev/stdout", "w")
                            except Exception as e:
                                logging.error("【An error occurred】: %s %s", e, rep_path)