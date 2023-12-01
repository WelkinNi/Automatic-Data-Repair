import pandas as pd
import numpy as np
from tqdm import tqdm
import jaro
import time
import argparse
import sys
import copy
import signal
from datetime import datetime
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


class Unified():
    def __init__(self, csv_d, csv_c, fd_path, theta=0.01, beta=0.4):
        self.resolve_fd(fd_path)
        self.dirty_csv = pd.read_csv(csv_d).astype(str)
        self.dirty_csv.fillna("nan", inplace=True)
        # self.dirty_csv = self.dirty_csv.iloc[:1000]
        self.theta = theta
        self.beta = beta
        self.clean_csv = pd.read_csv(csv_c).astype(str)
        self.clean_csv.fillna("nan", inplace=True)
        # self.clean_csv = self.clean_csv.iloc[:1000]
        self.rep_cells = []
        self.wrong_cells = []
        self.clean_in_cands = []
        self.clean_in_cands_repair_right = []
        self.out_csv = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Repaired_res/Unified/" + task_name[:-1] + "/repaired_" + task_name + check_string(dirty_path.split("/")[-1]) + ".csv"
        for i in range(len(self.dirty_csv)):
            for j in range(len(self.dirty_csv.columns)):
                if self.dirty_csv.iloc[i, j] != self.clean_csv.iloc[i, j]:
                    self.wrong_cells.append((i, j))

    def resolve_fd(self, fd_path):
        ori_f = open(fd_path)
        lines = ori_f.readlines()
        self.ori_fd = {}
        fd_num = 0
        for line in lines:
            fd_this = []
            line = line.split("⇒", 1)
            left, right = [i.strip() for i in line[0].strip().split(",")], \
                [i.strip() for i in line[1].strip().split(",")]
            fd_this.append(left)
            fd_this.append(right)
            self.ori_fd[fd_num] = fd_this
            fd_num = fd_num + 1

    def compute_core_deviant_patterns(self):
        self.orifd_core_pat = {}
        self.orifd_devi_pat = {}
        for idx, val in self.ori_fd.items():
            fd_df = self.dirty_csv[self.extract_attrs(idx)]
            idx_core, idx_devi = self.com_single_core_devi(idx, fd_df) 
            self.orifd_core_pat.update(idx_core), self.orifd_devi_pat.update(idx_devi)

    def com_single_core_devi(self, idx, fd_df):
        orifd_core_pat = {}
        orifd_devi_pat = {}
        orifd_core_pat[idx] = {}
        orifd_devi_pat[idx] = {}
        processed = {}
        length = len(fd_df)
        for i in range(length):
            fd_str = "\t".join(fd_df.iloc[i].astype(str).values.tolist())
            if fd_str in processed.keys():
                processed[fd_str].append(i)
            else:
                processed[fd_str] = []
                processed[fd_str].append(i)
        for key in processed.keys():
            if len(processed[key]) >= self.theta*length:
                orifd_core_pat[idx][key] = processed[key]
            else:
                orifd_devi_pat[idx][key] = processed[key]   
        return orifd_core_pat, orifd_devi_pat

    def FIND_DATA_REPAIRS(self, fd_idx):
        cost_f = 0
        #  self.pat_len_cal(fd_idx, self.orifd_core_pat) + self.pat_len_cal(fd_idx, self.orifd_devi_pat)
        core_values = [i.strip().split("\t") for i in self.orifd_core_pat[fd_idx].keys()]
        impact_fds = []
        attrs_list = self.extract_attrs(fd_idx)
        self.tmp_csv = copy.deepcopy(self.dirty_csv)
        
        for idx, _ in self.ori_fd.items():
            attrs = self.extract_attrs(idx)
            for attr in attrs:
                if attr in attrs_list:
                    impact_fds.append(idx)
                    break
        impact_fds.remove(fd_idx)
        self.orifd_core_pat_backup = copy.deepcopy(self.orifd_core_pat)
        self.orifd_devi_pat_backup = copy.deepcopy(self.orifd_devi_pat)
        cur_rep_store = []
        print("\nProcessing FD:" + str(fd_idx) + "Deviant Pattern")
        for devi_v in tqdm(self.orifd_devi_pat[fd_idx].keys(), ncols=90):
            cost_d = 0
            cur_devi_rep = []
            devi_value = devi_v.strip().split("\t")
            cost_d, best_rep, cands = self.get_best_repair(devi_value, core_values)
            # If best_rep is empty, then all possible corrections shoule be ignored
            if not best_rep:
                continue
            cost_d += 1
            for devi_idx in self.orifd_devi_pat[fd_idx][devi_v]:
                for attr in attrs_list:
                    if (devi_idx, list(self.dirty_csv.columns).index(attr)) not in self.clean_in_cands:
                        if self.clean_csv.iloc[devi_idx, list(self.dirty_csv.columns).index(attr)] in cands:
                            self.clean_in_cands.append((devi_idx, list(self.dirty_csv.columns).index(attr)))
                            if best_rep[attrs_list.index(attr)] == self.clean_csv.iloc[devi_idx, list(self.dirty_csv.columns).index(attr)]:
                                self.clean_in_cands_repair_right.append((devi_idx, list(self.dirty_csv.columns).index(attr)))
                    cur_devi_rep.append((devi_idx, list(self.dirty_csv.columns).index(attr)))
                    if not PERFECTED:
                        self.tmp_csv.iloc[devi_idx, list(self.dirty_csv.columns).index(attr)] = best_rep[attrs_list.index(attr)]
                    else:
                        if (devi_idx, list(self.dirty_csv.columns).index(attr)) not in self.wrong_cells:
                            continue
            if fd_idx in impact_fds:
                impact_fds.remove(fd_idx)
            # Cal impact cost
            for idx in impact_fds:
                changed_core_pat, changed_devi_pat = self.com_single_core_devi(idx, self.tmp_csv[attrs_list])
                cost_d += self.pat_len_changed_cal(idx, changed_core_pat, self.orifd_core_pat) +\
                    self.pat_len_changed_cal(idx, changed_devi_pat, self.orifd_devi_pat)
            # Cal current fd change
            changed_fdcore_pat, changed_fddevi_pat = self.com_single_core_devi(fd_idx, self.tmp_csv[attrs_list])
            delta_dl = self.pat_len_changed_cal(fd_idx, changed_fdcore_pat, self.orifd_core_pat) +\
                        self.pat_len_changed_cal(fd_idx, changed_fddevi_pat, self.orifd_devi_pat)
            impact_fds.append(fd_idx)
            if delta_dl < 0:
                self.compute_core_deviant_patterns()
                impact_fds = list(set(impact_fds))
                for idx in impact_fds:
                    # print(str(len(impact_fds)))
                    fd_df = self.tmp_csv[self.extract_attrs(idx)]
                    idx_core, idx_devi = self.com_single_core_devi(idx, fd_df) 
                    self.orifd_core_pat.update(idx_core), self.orifd_devi_pat.update(idx_devi)
                cost_f += cost_d
                cur_rep_store.extend(cur_devi_rep)
            else:
                for devi_idx in self.orifd_devi_pat[fd_idx][devi_v]:
                    for attr in attrs_list:
                        self.tmp_csv.iloc[devi_idx, list(self.dirty_csv.columns).index(attr)] = self.dirty_csv.iloc[devi_idx, list(self.dirty_csv.columns).index(attr)]
        cost_f += self.pat_len_cal(fd_idx, self.orifd_core_pat) + self.pat_len_cal(fd_idx, self.orifd_devi_pat)
        return cost_f, cur_rep_store

    def get_best_repair(self, devi_value, core_values):
        min = 1000
        best_core = []
        cands = []
        for core_sig in core_values:
            cost = 0
            change_idx = []
            if len(set(devi_value) & set(core_sig))/len(core_sig) < self.beta:
                continue
            for i in range(len(core_sig)):
                cands.append(core_sig[i])
                if core_sig[i] != devi_value[i]:
                    if jaro.jaro_winkler_metric(core_sig[i], devi_value[i]) == 0:
                        cost = cost + 1
                    else:
                        cost += jaro.jaro_winkler_metric(core_sig[i], devi_value[i])
                    change_idx.append(i) 
            if cost < min:
                min = cost
                best_core = core_sig
            else:
                continue
        return min, best_core, cands
            
    def pat_len_cal(self, fd_idx, pat):
        if len(pat) == 0 or len(pat[fd_idx]) == 0:
            return 0
        else: 
            cost = len(pat[fd_idx])*len(list(pat[fd_idx])[0].split("\t"))
        return cost

    def pat_len_changed_cal(self, fd_idx, pat_aft, pat_bef):
        cost = self.pat_len_cal(fd_idx, pat_aft) - self.pat_len_cal(fd_idx, pat_bef)
        return cost

    def COMPUTE_ICF(self):
        icf_score = {}
        for idx, val in self.ori_fd.items():
            left, right = self.ori_fd[idx][0], self.ori_fd[idx][1]
            fd_df = self.dirty_csv[left]
            processed = {}
            for i in range(len(self.dirty_csv)):
                fd_lstr = "\t".join(fd_df.iloc[i].astype(str).values.tolist())
                fd_rstr = "\t".join(self.dirty_csv[right].iloc[i].astype(str).values.tolist())
                if fd_lstr in processed.keys():
                    processed[fd_lstr].append(fd_rstr)
                else:
                    processed[fd_lstr] = []
                    processed[fd_lstr].append(fd_rstr)
            vio_num = 0
            for k, r_value in processed.items():
                if len(list(set(list(r_value)))) > 1:
                    vio_num += len(list(set(r_value))) - 1
            icf_score[idx] = vio_num / len(self.dirty_csv)
        return icf_score

    def COMPUTE_CONFLICT_SCORE(self):
        cf_score = {}
        for idx_i in range(len(self.ori_fd)):
            attrs_i = self.extract_attrs(idx_i)
            cf_score[idx_i] = 0
            for idx_j in range(len(self.ori_fd)):
                if idx_j == idx_i:
                    continue
                cf_num = 0
                attrs_j = self.extract_attrs(idx_j)
                for attr_i in attrs_i:
                    if attr_i in attrs_j:
                        cf_num += 1
                cf_score[idx_i] += cf_num/max(len(attrs_i), len(attrs_j))
            cf_score[idx_i] = cf_score[idx_i]/len(self.ori_fd)
        return cf_score

    def Cal_FD_impact(self, fd_idx):
        attrs_idx = self.extract_attrs(fd_idx)
        impact_fds = []
        for idx, val in self.ori_fd.items():
            attrs = self.extract_attrs(idx)
            for attr in attrs:
                if attr in attrs_idx:
                    impact_fds.append(idx)
        impact_fds = list(set(list(impact_fds)))
        impact_fds.remove(fd_idx)
        return impact_fds

    def extract_attrs(self, fd_idx):
        attrs = []
        attrs.extend(self.ori_fd[fd_idx][0])
        attrs.extend(self.ori_fd[fd_idx][1])
        return attrs
    
    def FIND_CONSTRAINT_REPAIRS(self, fd_idx):
        best_score = 10000
        winning_attr = -1
        fd_attrs = self.extract_attrs(fd_idx)
        cols = list(self.dirty_csv)
        coores_comp = -1
        for col in cols:
            if col not in fd_attrs:
                # Compute Score -> Related to Algorithm2
                homo = self.condition_ent_CK(fd_idx, col)
                com = self.condition_ent_KC(fd_idx, col)
                if homo < best_score:
                    winning_attr = col
                    best_score = homo
                    coores_comp = com
                elif abs(homo-best_score) < 1e-8:
                    if com > coores_comp:
                        winning_attr = col
                        best_score = homo
                        coores_comp = com
        attrs = self.extract_attrs(fd_idx)
        attrs.append(winning_attr)
        updated_df = self.dirty_csv[attrs]
        updfd_core_pat, updfd_devi_pat = self.com_single_core_devi(0, updated_df)
        cost = self.pat_len_cal(0, updfd_core_pat) + self.pat_len_cal(0, updfd_devi_pat)
        return cost, winning_attr

    def condition_ent_CK(self, fd_idx, attr):
        ent = 0
        left_len = len(self.ori_fd[fd_idx][0])
        right = len(self.ori_fd[fd_idx][1])
        fd_attrs = list(set(self.extract_attrs(fd_idx)))
        if attr in fd_attrs:
            fd_attrs.remove(attr)
        fd_attrs.append(attr)
        attr_cal = {}
        fd_attr_cal = {}
        all_cal = {}
        refer_df = self.dirty_csv[fd_attrs]
        df_cols = list(refer_df.columns)
        for i in range(len(refer_df)):
            attr_val = refer_df.iloc[i, df_cols.index(attr)]
            fd_attr_val = "\t".join(refer_df[fd_attrs[:-1]].iloc[i].astype(str).values.tolist())
            all_val = "\t".join(refer_df.iloc[i].astype(str).values.tolist())
            if attr_val not in fd_attr_cal.keys():
                attr_cal[attr_val] = 1
                fd_attr_cal[attr_val] = {}
                fd_attr_cal[attr_val][fd_attr_val] = 1
            else:
                if fd_attr_val in fd_attr_cal[attr_val].keys():
                    fd_attr_cal[attr_val][fd_attr_val] += 1
                else:
                    fd_attr_cal[attr_val][fd_attr_val] = 1
                attr_cal[attr_val] += 1
            if all_val not in all_cal.keys():
                all_cal[all_val] = 1
            else:
                all_cal[all_val] += 1
        for attr_v in fd_attr_cal.keys():
            # the same attr on the both terms of fd should be avoided
            left2right = {}
            for fd_v in fd_attr_cal[attr_v].keys():
                attr_ent = 0
                left_v = "\t".join(fd_v.split("\t")[0:left_len])
                all_v = fd_v + "\t" + str(attr_v)
                attr_ent += - all_cal[all_v]/len(refer_df) * np.log10(fd_attr_cal[attr_v][fd_v]/attr_cal[attr_v])
                if left_v in left2right.keys():
                    left2right[left_v].append("\t".join(fd_v.split("\t")[left_len:]))
                else:
                    left2right[left_v] = []
                    left2right[left_v].append("\t".join(fd_v.split("\t")[left_len:]))
                if len(set(left2right[left_v])) == 1:
                    ent += attr_ent
        return ent

    def condition_ent_KC(self, fd_idx, attr):
        ent = 0
        fd_attrs = list(set(self.extract_attrs(fd_idx)))
        if attr in fd_attrs:
            fd_attrs.remove(attr)
        fd_attrs.append(attr)
        fd_cal = {}
        attr_fd_cal = {}
        all_cal = {}
        refer_df = self.dirty_csv[fd_attrs]
        df_cols = list(refer_df.columns)
        for i in range(len(refer_df)):
            attr_val = refer_df.iloc[i, df_cols.index(attr)]
            fd_attr_val = "\t".join(refer_df[fd_attrs[:-1]].iloc[i].astype(str).values.tolist())
            all_val = "\t".join(refer_df.iloc[i].astype(str).values.tolist())
            if fd_attr_val not in attr_fd_cal.keys():
                fd_cal[fd_attr_val] = 1
                attr_fd_cal[fd_attr_val] = {}
                attr_fd_cal[fd_attr_val][attr_val] = 1
            else:
                if attr_val in attr_fd_cal[fd_attr_val]: 
                    attr_fd_cal[fd_attr_val][attr_val] += 1
                else:
                    attr_fd_cal[fd_attr_val][attr_val] = 1
                fd_cal[fd_attr_val] += 1
            if all_val not in all_cal.keys():
                all_cal[all_val] = 1
            else:
                all_cal[all_val] += 1
        for fd_v in attr_fd_cal.keys():
            for attr_v in attr_fd_cal[fd_v].keys():
                all_v = fd_v + "\t" + str(attr_v)
                ent += - all_cal[all_v]/len(refer_df) * np.log10(attr_fd_cal[fd_v][attr_v]/fd_cal[fd_v])
                a = []
        return ent

    def run(self):
        com_score = {}
        print("Computing Core and Deviant  Patterns")
        for fd_idx in tqdm(range(len(self.ori_fd)), ncols=90):
            self.compute_core_deviant_patterns()
        icf_list = self.COMPUTE_ICF()
        confl_list = self.COMPUTE_CONFLICT_SCORE()
        for fd_idx in range(len(self.ori_fd)):
            com_score[fd_idx] = 1/2*(icf_list[fd_idx] + confl_list[fd_idx])
        FList = sorted(com_score.items(), key=lambda item: item[1], reverse=True)
        for fd_idx, _ in tqdm(FList,ncols=90):
            cost_d, cur_rep = self.FIND_DATA_REPAIRS(fd_idx)
            cost_c, winning_attr = self.FIND_CONSTRAINT_REPAIRS(fd_idx)
            print("FD_idx: {idx}, Data Repair Cost: {dcost}, Rule Repair Cost: {rcost}".format(idx=fd_idx, dcost=cost_d, rcost=cost_c))
            if cost_d < cost_c:
                # applied data repair
                self.dirty_csv = copy.deepcopy(self.tmp_csv)
                self.rep_cells.extend(cur_rep)
            else:
                # data repair recover
                self.orifd_core_pat = copy.deepcopy(self.orifd_core_pat_backup)
                self.orifd_devi_pat = copy.deepcopy(self.orifd_devi_pat_backup)
                # applied fd repair
                self.ori_fd[fd_idx][0].append(winning_attr)
                core_p, devi_p = self.com_single_core_devi(fd_idx, self.dirty_csv[self.extract_attrs(fd_idx)])
                self.orifd_core_pat.update(core_p), self.orifd_devi_pat.update(devi_p)
        self.evaluation()
        
    def evaluation(self):
        self.rep_cells = list(set(self.rep_cells))
        self.wrong_cells = list(set(self.wrong_cells))
        self.repair_right_cells = []
        
        if True:
            if not PERFECTED:
                self.rep_cells = list(set(self.rep_cells))
                self.wrong_cells = list(set(self.rep_cells))
                det_right = 0
                out_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Exp_result/Unified/" + task_name[:-1] +"/onlyED_" + task_name + check_string(dirty_path.split("/")[-1]) + ".txt"
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

                out_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Exp_result/Unified/" + task_name[:-1] +"/oriED+EC_" + task_name + check_string(dirty_path.split("/")[-1]) + ".txt"
                res_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Repaired_res/Unified/" + task_name[:-1] + "/repaired_" + task_name + check_string(dirty_path.split("/")[-1]) + ".csv"
                self.dirty_csv.to_csv(res_path, index=False)
                f = open(out_path, 'w')
                sys.stdout = f
                end_time = time.time()
                rep_right = 0
                rep_total = len(self.rep_cells)
                wrong_cells = len(self.wrong_cells)
                rec_right = 0
                for cell in self.rep_cells:
                    if self.dirty_csv.iloc[cell[0], cell[1]] == self.clean_csv.iloc[cell[0], cell[1]]:
                        rep_right += 1
                        self.repair_right_cells.append(cell)
                for cell in self.wrong_cells:
                    if self.dirty_csv.iloc[cell[0], cell[1]] == self.clean_csv.iloc[cell[0], cell[1]]:
                        rec_right += 1
                pre = rep_right / (rep_total+1e-10)
                rec = rec_right / (wrong_cells+1e-10)
                f1 = 2*pre*rec / (rec+pre+1e-10)
                print("{pre}\n{rec}\n{f1}\n{time}".format(pre=pre, rec=rec, f1=f1, time=(end_time-start_time)))
                f.close()

                sys.stdout = sys.__stdout__
                out_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Exp_result/Unified/" + task_name[:-1] + "/all_compute_" + task_name + check_string(dirty_path.split("/")[-1]) + ".txt"
                f = open(out_path, 'w')
                sys.stdout = f
                right2wrong = 0
                right2right = 0
                wrong2right = 0
                wrong2wrong = 0
                
                self.clean_in_cands = [cell for cell in self.clean_in_cands if cell in self.rep_cells]
                self.clean_in_cands_repair_right = [cell for cell in self.clean_in_cands_repair_right if cell in self.rep_cells]

                # for cell in self.clean_in_cands:
                #     if cell not in self.rep_cells:
                #         self.clean_in_cands.remove(cell)
                # for cell in self.clean_in_cands_repair_right:
                #     if cell not in self.rep_cells:
                #         self.clean_in_cands_repair_right.remove(cell)

                rep_total = len(self.rep_cells)
                wrong_cells = len(self.wrong_cells)
                for cell in self.repair_right_cells:
                    if cell in self.wrong_cells:
                        wrong2right += 1
                    else:
                        right2right += 1
                print("rep_right:"+str(rep_right))
                print("rec_right:"+str(rec_right))
                print("wrong_cells:"+str(wrong_cells))
                print("prec:"+str(pre))
                print("rec:"+str(rec))
                print("wrong2right:"+str(wrong2right))
                print("right2right:"+str(right2right))
                self.repair_wrong_cells = [i for i in self.rep_cells if i not in self.repair_right_cells]
                for cell in self.repair_wrong_cells:
                    if cell in self.wrong_cells:
                        wrong2wrong += 1
                    else:
                        right2wrong += 1
                print("wrong2wrong:"+str(wrong2wrong))
                print("right2wrong:"+str(right2wrong))
                print("proportion of clean value in candidates:"+str(len(self.clean_in_cands)/(rep_total+1e-8)))
                print("proportion of clean value in candidates and selected correctly:"+str(len(self.clean_in_cands_repair_right)/(len(self.clean_in_cands)+1e-8)))
                f.close()
            else:
                out_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Exp_result/Unified/" + task_name[:-1] +"/prefectED+EC_" + task_name + check_string(dirty_path.split("/")[-1]) + ".txt"
                res_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Repaired_res/Unified/" + task_name[:-1] + "/perfect_repaired_" + task_name + check_string(dirty_path.split("/")[-1]) + ".csv"
                self.dirty_csv.to_csv(res_path, index=False)
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

if __name__ == "__main__":
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
    
    # dirty_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/data with dc_rules/flights/noise/flights-outer_error-10.csv"
    # rule_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/data with dc_rules/flights/dc_rules-validate-fd-horizon.txt"
    # clean_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/data with dc_rules/flights/clean.csv"
    # task_name = "flights1"
    # out_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Exp_result/Unified/" + task_name[:-1] +"/onlyED_" + task_name + check_string(dirty_path.split("/")[-1]) + dirty_path[-6:-4] + ".txt"
    # ONLYED = 1
    # PERFECTED = 0

    time_limit = 96*3600
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(time_limit)
    try:
        start_time = time.time()
        U_Clean = Unified(dirty_path, clean_path, rule_path, theta=0.001, beta=0.4)
        U_Clean.run()
    except TimeoutError as e: 
        print("Time exceeded:", e, task_name, dirty_path)
        out_file = open("/data/nw/DC_ED/References_inner_and_outer/DATASET/aggre_results/timeout_log.txt", "a")
        now = datetime.now()
        out_file.write(now.strftime("%Y-%m-%d %H:%M:%S"))
        out_file.write("Time out: Unified.py ")
        out_file.write(f" {task_name}")
        out_file.write(f" {dirty_path}\n")
        out_file.close()