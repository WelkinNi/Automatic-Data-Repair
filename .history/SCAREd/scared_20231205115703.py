import pandas as pd
import numpy as np
import signal
import copy
import time
import sys
import os
import raha
import argparse
import shutil
import logging
from collections import Counter
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import warnings
from tqdm import tqdm
from rich.progress import track
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
warnings.filterwarnings("ignore")
import re

def check_string(string: str):
    if re.search(r"-inner_error-", string):
        return "-inner_error-" + string[-6:-4]
    elif re.search(r"-outer_error-", string):
        return "-outer_error-" + string[-6:-4]
    elif re.search(r"-inner_outer_error-", string):
        return "-inner_outer_error-" + string[-6:-4]
    elif re.search(r"-dirty-original_error-", string):
        return "-original_error-" + string[-9:-4]

def handler(signum, frame):
    raise TimeoutError("Time exceeded")

class SCAREd:
    def __init__(self, csv_d, csv_c, reliable_attrs=[]):
        self.dirty_csv = pd.read_csv(csv_d).astype(str)
        self.dirty_csv.fillna("null")
        
        self.rep_csv = copy.deepcopy(self.dirty_csv)
        self.schema = list(self.dirty_csv.columns)
        self.clean_csv = pd.read_csv(csv_c).astype(str)
        self.clean_csv.fillna("null")
        self.rep_cells = []
        self.wrong_cells = []
        # self.out_csv = csv_d[:-4] + "-relatrust_cleaned.csv"
        # self.tau = 0.01*len(self.schema)*len(self.dirty_csv)
        self.reliable_attrs = reliable_attrs
        self.flexible_attrs = []
        # self.dirty_csv = self.dirty_csv.iloc[:200]
        # self.clean_csv = self.clean_csv.iloc[:200]
        for i in range(len(self.dirty_csv)):
            for j in range(1, len(self.dirty_csv.columns)):
                if self.dirty_csv[self.dirty_csv.columns[j]][i] != self.clean_csv[self.dirty_csv.columns[j]][i]:
                    self.wrong_cells.append((i, j))
        self.dirty_csv.insert(0, 'Index', list(range(len(self.dirty_csv))))
        self.rep_csv.insert(0, 'Index', list(range(len(self.rep_csv))))
        self.rs = {}
        det_count = Counter([key[1] for key in detection_dictionary.keys()])
        re_attrs = list(sorted(det_count.items(), key = lambda kv: kv[1], reverse=False)[:2])
        re_attrs = [self.clean_csv.columns[i[0]] for i in re_attrs]
        self.reliable_attrs.append('Index')
        self.reliable_attrs.extend(re_attrs)

    def find_order(self):
        return self.schema
        
    def partition(self):
        # col_card = {}
        # for i in self.schema:
        #     col_card[i] = len(self.dirty_csv[i].value_counts())
        # part_attrs = []
        # if len(self.reliable_attrs) > 0:
        #     part_attrs = copy.deepcopy(self.reliable_attrs)
        # else:
        #     ver_edg_cnt = sorted(col_card.items(), key=lambda kv: kv[1], reverse=True)
        #     part_attrs.append(ver_edg_cnt[0][0])
        #     part_attrs.append(ver_edg_cnt[1][0])
        #     part_attrs.append(ver_edg_cnt[2][0])
        #     self.reliable_attrs.append('Index')
        #     self.reliable_attrs.extend(copy.deepcopy(part_attrs))
        self.flexible_attrs = [attr for attr in self.dirty_csv.columns if attr not in self.reliable_attrs]
        df_partitions = []
        for attr in self.reliable_attrs:
            attr_card = list(dict(self.dirty_csv[attr].value_counts()).keys())
            df_partition = []
            for val in attr_card:
                df_partition.append(self.dirty_csv[self.dirty_csv[attr] == val])
            df_partitions.append(df_partition)
        return df_partitions

    def get_model(self, data):
        models = {}
        r_attrs = copy.deepcopy(self.reliable_attrs)
        f_attrs = copy.deepcopy(self.flexible_attrs)
        for i in range(len(f_attrs)):
            model = {}
            enc_x = OneHotEncoder(handle_unknown='ignore', sparse=False)
            enc_y = LabelEncoder()
            model['xenc'] = enc_x
            model['yenc'] = enc_y
            X = enc_x.fit_transform(data[r_attrs].values)
            y = enc_y.fit_transform(data[f_attrs[i]].values)
            clf = MultinomialNB(fit_prior=True)
            clf.fit(X, y)
            model['model'] = clf
            r_attrs.append(f_attrs[i])
            models[i] = model
        return models

    def run(self):
        df_partitions = self.partition()
        for i in range(len(df_partitions)):
            print("Candidates Storage Generation:" + str(i+1) + "/" + str(len(df_partitions)))
            for j in tqdm(range(len(df_partitions[i])), ncols=80):
                models = self.get_model(df_partitions[i][j])
                self.get_all_preds(df_partitions[i][j], models, i, j)
        print("Conduct Repair")
        for i in tqdm(range(len(self.dirty_csv)), ncols=80):
            assign = self.rs[i]
            fin_select = self.get_final_pred(assign)
            for key in fin_select.keys():
                self.rep_csv.iloc[i, list(self.rep_csv.columns).index(key)] = fin_select[key]
                if self.dirty_csv.iloc[i, list(self.rep_csv.columns).index(key)] != fin_select[key]:
                    self.rep_cells.append((i, list(self.clean_csv.columns).index(key)))
        self.evaluation()

        # print("Finish Repairing Data")
    def evaluation(self):
        self.rep_cells = list(set(self.rep_cells))
        self.wrong_cells = list(set(self.wrong_cells))
        
        if True:
            if not PERFECTED:
                det_right = 0
                out_path = "./Exp_result/scared/" + task_name[:-1] + "/onlyED_" + task_name + check_string(dirty_path.split("/")[-1]) + ".txt"
                f = open(out_path, 'w')
                sys.stdout = f
                end_time = time.time()
                for cell in self.rep_cells:
                    if cell in self.wrong_cells:
                        det_right = det_right + 1
                pre = det_right / (len(self.rep_cells)+1e-10)
                rec = det_right / (len(self.wrong_cells)+1e-10)
                f1 = 2*pre*rec/(pre+rec+1e-10)
                print("{pre}\n{rec}\n{f1}\n{time}".format(pre=pre, rec=rec, f1=f1, time=(end_time-start_time)))
                f.close()

                out_path = "./Exp_result/scared/" + task_name[:-1] + "/oriED+EC_" + task_name + check_string(dirty_path.split("/")[-1]) + ".txt"
                res_path = "./Repaired_res/scared/" + task_name[:-1] + "/repaired_" + task_name + check_string(dirty_path.split("/")[-1]) + ".csv"
                self.rep_csv.drop('Index', axis=1, inplace=True)
                self.rep_csv.to_csv(res_path, index=False, columns=list(self.rep_csv.columns))
                f = open(out_path, 'w')
                sys.stdout = f
                end_time = time.time()
                rep_right = 0
                rep_total = len(self.rep_cells)
                wrong_cells = len(self.wrong_cells)
                rec_right = 0
                for cell in self.rep_cells:
                    if self.rep_csv.iloc[cell[0], cell[1]] == self.clean_csv.iloc[cell[0], cell[1]]:
                        rep_right += 1
                for cell in self.wrong_cells:
                    if self.rep_csv.iloc[cell[0], cell[1]] == self.clean_csv.iloc[cell[0], cell[1]]:
                        rec_right += 1
                pre = rep_right / (rep_total+1e-10)
                rec = rec_right / (wrong_cells+1e-10)
                f1 = 2*pre*rec / (rec+pre+1e-10)
                print("{pre}\n{rec}\n{f1}\n{time}".format(pre=pre, rec=rec, f1=f1, time=(end_time-start_time)))
                f.close()
            else:
                out_path = "./Exp_result/scared/" + task_name[:-1] + "/perfectED+EC_" + task_name + check_string(dirty_path.split("/")[-1]) + ".txt"
                res_path = "./Repaired_res/scared/" + task_name[:-1] + "/perfect_repaired_" + task_name + check_string(dirty_path.split("/")[-1]) + ".csv"
                self.rep_csv.to_csv(res_path, index=False, columns=list(self.rep_csv.columns))
                f = open(out_path, 'w')
                sys.stdout = f
                end_time = time.time()
                rep_right = 0
                rep_total = len(self.rep_cells)
                wrong_cells = len(self.wrong_cells)
                rec_right = 0
                rep_t = 0
                for cell in self.wrong_cells:
                    if cell in self.rep_cells:
                        rep_t += 1
                        if self.dirty_csv.iloc[cell[0], cell[1]] == self.clean_csv.iloc[cell[0], cell[1]]:
                            rec_right += 1
                pre = rec_right / (rep_t+1e-10)
                rec = rec_right / (wrong_cells+1e-10)
                f1 = 2*pre*rec / (rec+pre+1e-10)
                print("{pre}\n{rec}\n{f1}\n{time}".format(pre=pre, rec=rec, f1=f1, time=(end_time-start_time)))
                f.close()

    def get_final_pred(self, assign):
        attr_val_edge = {}
        attr_val_val = {}
        for i in range(len(self.flexible_attrs)):
            attr_val_edge[self.flexible_attrs[i]] = {}
            attr_val_val[self.flexible_attrs[i]] = {}
        for key in assign.keys():
            for i in range(len(self.flexible_attrs)):
                data = assign[key][1][self.flexible_attrs]
                attr_i = self.flexible_attrs[i]
                val_i = data[attr_i]
                node_i = (attr_i, val_i)
                # if val_i not in attr_val_edge[attr_i].keys():
                #     attr_val_edge[attr_i][val_i] = {}
                for j in range(i+1, len(self.flexible_attrs)):
                    attr_j = self.flexible_attrs[j]
                    val_j = data[attr_j]
                    node_j = (attr_j, val_j)
                    # attr_val_edge[attr_i][val_i][node_j] = assign[key][2]
                    if val_i not in attr_val_edge[attr_i].keys():
                        attr_val_edge[attr_i][val_i] = {}
                        attr_val_edge[attr_i][val_i][node_j] = assign[key][2]
                    else:
                        if node_i in attr_val_edge[attr_i][val_i].keys():
                            attr_val_edge[attr_i][val_i][node_j] += assign[key][2]
                        else:
                            attr_val_edge[attr_i][val_i][node_j] = assign[key][2]

                    if val_j not in attr_val_edge[attr_j].keys():
                        attr_val_edge[attr_j][val_j] = {}
                        attr_val_edge[attr_j][val_j][node_i] = assign[key][2]
                    else:
                        if node_i in attr_val_edge[attr_j][val_j].keys():
                            attr_val_edge[attr_j][val_j][node_i] += assign[key][2]
                        else:
                            attr_val_edge[attr_j][val_j][node_i] = assign[key][2]
        for attr in attr_val_edge.keys():
            for val in attr_val_edge[attr].keys():
                attr_val_val[attr][val] = 0
                for node in attr_val_edge[attr][val].keys():
                    attr_val_val[attr][val] = attr_val_val[attr][val] + attr_val_edge[attr][val][node]
        final_select = self.khs_solution(attr_val_val, attr_val_edge)
        update = {}
        for key in final_select.keys():
            update[key[0]] = key[1]
        return update

    def khs_solution(self, attr_val_val, attr_val_edge):
        attr_val_dict = {}
        for attr in attr_val_val.keys():
            for val in attr_val_val[attr].keys():
                node = (attr, val)
                attr_val_dict[node] = attr_val_val[attr][val]
        while not self.end_condition(attr_val_edge):
            sorted_node = sorted(attr_val_dict.items(), key=lambda kv:(kv[1], kv[0]))
            del_node = None
            for node in sorted_node:
                if len(attr_val_val[node[0][0]]) == 1:
                    continue
                else:
                    del_node = copy.deepcopy(node[0])
                    break
            if del_node is not None:
                for node in attr_val_edge[del_node[0]][del_node[1]]:
                    attr_val_dict[(node[0], node[1])] = attr_val_dict[(node[0], node[1])] - attr_val_edge[node[0]][node[1]][(del_node[0], del_node[1])]
                    del attr_val_edge[node[0]][node[1]][(del_node[0], del_node[1])]
                del attr_val_edge[del_node[0]][del_node[1]]
                del attr_val_dict[(del_node[0], del_node[1])]
        return attr_val_dict

    def end_condition(self, attr_val_edge):
        flag = 1
        for key in attr_val_edge:
            if len(attr_val_edge[key]) != 1:
                flag = 0
        return flag

    def get_ori_prob(self, data, i, ori_prob, models, r_set, f_set, k_cur):
        if len(f_set) == k_cur:
            return ori_prob
        f_attr = f_set[k_cur]
        r_data, f_data = data[r_set], data[f_attr]
        x_data = r_data.iloc[i]
        y_data = f_data.iloc[i]
        y_data = models[k_cur]['yenc'].transform(np.array([y_data]))
        y_prob = models[k_cur]['model'].predict_proba(models[k_cur]['xenc'].transform(x_data.values.reshape(1,-1)))[0][y_data[0]] * ori_prob
        k_cur = k_cur + 1
        r_set_temp = copy.deepcopy(r_set)
        r_set_temp.append(f_attr)
        prob = self.get_ori_prob(data, i, y_prob, models, r_set_temp, f_set, k_cur)
        return prob
    
    def get_all_preds(self, data, models, i_idx, j_idx):
        for i in range(len(data)):
            idx = data[self.reliable_attrs].iloc[i, 0]
            if idx not in self.rs:
                self.rs[idx] = {}
            tuple_prob = self.get_ori_prob(data, i, 1.0, models, self.reliable_attrs, self.flexible_attrs, 0)
            self._get_single_preds(data, i, 1.0, models, self.reliable_attrs, self.flexible_attrs, 0, i_idx, j_idx, tuple_prob)

    def _get_single_preds(self, data, i, ori_prob, models, r_set, f_set, k_cur, i_idx, j_idx, tuple_prob):
        if len(f_set) == k_cur:
            com_prob = tuple_prob 
            ori_prob = ori_prob*len(data)/len(self.clean_csv)
            storage = (com_prob, data[r_set].iloc[i], ori_prob)
            idx = data[r_set].iloc[i, 0]
            if (i_idx, j_idx) in self.rs[idx].keys():
                if ori_prob > self.rs[idx][(i_idx, j_idx)][2]:
                    self.rs[idx][(i_idx, j_idx)] = storage
            else:
                self.rs[idx][(i_idx, j_idx)] = storage
            return None
        f_attr = f_set[k_cur]
        r_data, f_data = data[r_set], data[f_attr]
        x_data = r_data.iloc[i]
        y_data = f_data.iloc[i]
        y_pred = models[k_cur]['model'].predict(models[k_cur]['xenc'].transform(x_data.values.reshape(1,-1)))
        y_prob = max(models[k_cur]['model'].predict_proba(models[k_cur]['xenc'].transform(x_data.values.reshape(1,-1)))[0])
        y_pred = models[k_cur]['yenc'].inverse_transform(y_pred)[0]
        r_set_temp = copy.deepcopy(r_set)
        r_set_temp.append(f_attr)
        # pre_list = list(models[k_cur]['yenc'].inverse_transform(models[k_cur]['model'].predict(models[k_cur]['xenc'].transform(r_data.values))))
        # act_list = f_data.values.reshape(-1,).tolist()
        # re_loss = sum(1 for i in range(len(pre_list)) if pre_list[i]==act_list[i]) / len(act_list)
        k_cur = k_cur + 1
        # y_prob = y_prob*(1)*len(data)/len(self.clean_csv)
        self._get_single_preds(data, i, y_prob, models, r_set_temp, f_set, k_cur, i_idx, j_idx, tuple_prob)
        if y_pred != y_data:
            data_copy = copy.deepcopy(data)
            if not PERFECTED:
                data_copy[f_attr].iloc[i] = y_pred
                # y_prob = y_prob*(1)*len(data)/len(self.clean_csv)
                self._get_single_preds(data_copy, i, y_prob, models, r_set_temp, f_set, k_cur, i_idx, j_idx, tuple_prob)
            else:
                if (i, list(self.clean_csv.columns).index(f_attr)) not in self.wrong_cells:
                    pass
                else:
                    data_copy[f_attr].iloc[i] = y_pred
                    self._get_single_preds(data_copy, i, y_prob, models, r_set_temp, f_set, k_cur, i_idx, j_idx, tuple_prob)
    
    def _model_quality(self, data, model, x_data, y_data):
        Re = len(data)/len(self.dirty_csv)
        y_pred = model['model'].predict(model['xenc'].transform(x_data.values.reshape(1,-1)))
        y_pred = model['yenc'].inverse_transform(y_pred)[0]
        return Re

    def _getCondEntropy(self, data, xname, yname):
        xs = data[xname].unique()
        ys = data[yname].unique()
        p_x = data[xname].value_counts() / data.shape[0]
        ce = 0
        for x in xs:
            ce += p_x[x]*self._getEntropy(data[data[xname] == x][yname])
            print(str(x) + ":" + str(p_x[x]*self._getEntropy(data[data[xname] == x][yname])))
        return ce
    
    def _getEntropy(self, data):
        if not isinstance(data, pd.core.series.Series):
            s = pd.Series(data)
        prt_ary = data.groupby(data).count().values / float(len(data))
        return sum(-(np.log2(prt_ary)*prt_ary))


if __name__ == "__main__": 
    time_limit = 24*3600
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(time_limit)
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--clean_path', type=str, default=None)
        parser.add_argument('--dirty_path', type=str, default=None)
        parser.add_argument('--rule_path', type=str, default=None)
        parser.add_argument('--task_name', type=str, default=None)
        parser.add_argument('--onlyed', type=int, default=None)
        parser.add_argument('--perfected', type=int, default=None)
        args = parser.parse_args()
        dirty_path = args.dirty_path
        clean_path = args.clean_path
        task_name = args.task_name
        rule_path = args.rule_path
        ONLYED = args.onlyed
        PERFECTED = args.perfected

        # dirty_path = "./data with dc_rules/hospital/noise/hospital-inner_outer_error-10.csv"
        # rule_path = "./data with dc_rules/tax/dc_rules-validate-fd-horizon.txt"
        # clean_path = "./data with dc_rules/hospital/clean.csv"
        # task_name = "hospital1"
        # ONLYED = 0
        # PERFECTED = 0

        detection_dictionary = {}
        if not PERFECTED:
            stra_path = "./data with dc_rules/" + task_name[:-1] + "/noise/raha-baran-results-" + 'scared'+task_name+check_string(dirty_path)
            if os.path.exists(stra_path):
                shutil.rmtree(stra_path)
            stra_path = "/data/nw/DC_ED/References/DATASET/data with dc_rules/" + task_name[:-1] + "/noise/raha-baran-results-" + 'scared'+task_name+check_string(dirty_path)
            if os.path.exists(stra_path):
                shutil.rmtree(stra_path)
            stra_path = "./data with dc_rules/tax/split_data/raha-baran-results-" + 'scared'+task_name+check_string(dirty_path)
            if os.path.exists(stra_path):
                shutil.rmtree(stra_path)
            stra_path = "./data with dc_rules/tax/split_data/raha-baran-results-" + 'scared'+task_name+check_string(dirty_path)
            if os.path.exists(stra_path):
                shutil.rmtree(stra_path)
            dataset_dictionary = {
                "name": 'scared'+task_name+check_string(dirty_path),
                "path": dirty_path,
                "clean_path": clean_path
            }
            start_time = time.time()
            app = raha.Detection()
            detection_dictionary = app.run(dataset_dictionary)
        else:
            clean_df = pd.read_csv(clean_path).astype(str)
            dirty_df = pd.read_csv(dirty_path).astype(str)
            clean_df = clean_df.fillna("nan")
            dirty_df = dirty_df.fillna("nan")
            for i in range(len(clean_df)):
                for j in range(len(clean_df.columns)):
                    if dirty_df.iloc[i, j] != clean_df.iloc[i, j]:
                        detection_dictionary[(i, j)] = 'dummy'

        start_time = time.time()
        logging.basicConfig(level=logging.DEBUG)
        scared_cleaner = SCAREd(dirty_path, clean_path)
        scared_cleaner.run()
    except TimeoutError as e: 
        print("Time exceeded:", e, task_name, dirty_path)
        out_file = open("./aggre_results/timeout_log.txt", "a")
        now = datetime.now()
        out_file.write(now.strftime("%Y-%m-%d %H:%M:%S"))
        out_file.write(" Scared.py: ")
        out_file.write(f" {task_name}")
        out_file.write(f" {dirty_path}\n")
        out_file.close()
