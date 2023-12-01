import sys
import os
path = '/data/nw/DC_ED/References_inner_and_outer/holoclean-master'
os.chdir(path)
sys.path.append(path)
import holoclean
import argparse
import time
import pandas as pd
from detect import NullDetector, ViolationDetector
from repair.featurize import *
from dataset import AuxTables
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

    # task = "hospital"
    # num = 90
    # dirty_path = "/data/nw/DC_ED/References/DATASET/data with dc_rules/"+str(task)+"/noise/"+str(task)+"-inner_outer_error-" + str(num) + ".csv"
    # clean_path = "/data/nw/DC_ED/References/DATASET/data with dc_rules/"+str(task)+"/clean.csv"
    # task_name = "hospital1"
    # rule_path = "/data/nw/DC_ED/References/DATASET/data with dc_rules/"+str(task)+"/dc_rules_holoclean.txt"
    # ONLYED = 0
    # PERFECTED = 0

    clean_df = pd.read_csv(clean_path).astype('str')
    dirty_df = pd.read_csv(dirty_path).astype('str')
    clean_df.fillna("nan", inplace=True)
    dirty_df.fillna("nan", inplace=True)
    clean_df = clean_df.apply(lambda col: col.str.lower())
    dirty_df = dirty_df.apply(lambda col: col.str.lower())

    wrong_cells = []
    for i in range(len(dirty_df)):
        for j in range(len(dirty_df.columns)):
            if dirty_df.iloc[i, j] != clean_df.iloc[i, j]:
                wrong_cells.append((i, j))

    start_time = time.time()
    # 1. Setup a HoloClean session.
    hc = holoclean.HoloClean(
        db_name='holo',
        domain_thresh_1=0,
        domain_thresh_2=0,
        weak_label_thresh=0.99,
        max_domain=10000,
        cor_strength=0.6,
        nb_cor_strength=0.8,
        epochs=10,
        weight_decay=0.01,
        learning_rate=0.001,
        threads=1,
        batch_size=1,
        verbose=True,
        timeout=3*60000,
        feature_norm=False,
        weight_norm=False,
        print_fw=True
    ).session

    # 2. Load training data and denial constraints.
    hc.load_data(task_name, dirty_path)
    hc.load_dcs(rule_path)
    hc.ds.set_constraints(hc.get_dcs())

    # # 3. Detect erroneous cells using these two detectors.
    # detectors = [NullDetector(), ViolationDetector()]
    # hc.detect_errors(detectors)

    # # 4. Repair errors utilizing the defined features.
    # hc.setup_domain()
    # featurizers = [
    #     InitAttrFeaturizer(),
    #     OccurAttrFeaturizer(),
    #     FreqFeaturizer(),
    #     ConstraintFeaturizer(),
    # ]

    # hc.repair_errors(featurizers)

    # # 5. Evaluate the correctness of the results.
    # hc.evaluate(fpath='/data/nw/DC_ED/References_inner_and_outer/holoclean-master/testdata/hospital_clean.csv',
    #             tid_col='tid',
    #             attr_col='attribute',
    #             val_col='correct_val')




    # 3. Detect erroneous cells using these two detectors.
    det_cells = []
    if not PERFECTED:
        detectors = [NullDetector(), ViolationDetector()]
        hc.detect_errors(detectors)
        det_df = hc.detect_engine.errors_df
        for i in range(len(det_df)):
            det_cells.append((det_df.iloc[i, 0], list(dirty_df).index(det_df.iloc[i, 1])))
    else:
        det_cells = wrong_cells
        errors_df = pd.DataFrame(columns=['_tid_', 'attribute'])
        print("errors_df")
        for cell in wrong_cells:
            res = [cell[0], dirty_df.columns[cell[1]]]
            errors_df.loc[len(errors_df), :] = res
        ds = hc.detect_engine.ds
        errors_df['_cid_'] = errors_df.apply(lambda x: ds.get_cell_id(x['_tid_'], x['attribute']), axis=1)
        hc.detect_engine.ds.generate_aux_table(AuxTables.dk_cells, errors_df, store=True)
        hc.detect_engine.ds.aux_table[AuxTables.dk_cells].create_db_index(ds.engine, ['_cid_'])

    det_cnt = 0
    for cell in det_cells:
        if cell in wrong_cells:
            det_cnt = det_cnt + 1
    det_pre = det_cnt/(len(det_cells)+1e-10)
    det_rec = det_cnt/(len(wrong_cells)+1e-10)
    det_f1 = 2*det_pre*det_rec/(det_pre+det_rec)

    # get detected error cells
    # hc.detect_engine.errors_df

    # 4. Repair errors utilizing the defined features.
    hc.setup_domain()
    featurizers = [
        InitAttrFeaturizer(),
        OccurAttrFeaturizer(),
        FreqFeaturizer(),
        ConstraintFeaturizer(),
    ]
    hc.repair_errors(featurizers)

    infer_val_df = hc.repair_engine.ds.repaired_vals
    domain = hc.domain_engine.domain

    repaired_df = hc.repair_engine.ds.repaired_data.df
    repaired_df = repaired_df.drop('_tid_', axis=1)

    rep_cnt = 0
    rep_all = 0
    for cell in det_cells:
        if repaired_df.iloc[cell[0], cell[1]] != dirty_df.iloc[cell[0], cell[1]]:
            rep_all = rep_all + 1
            if repaired_df.iloc[cell[0], cell[1]] == clean_df.iloc[cell[0], cell[1]]:
                rep_cnt = rep_cnt + 1
    rep_pre = rep_cnt/(rep_all+1e-10)
    rep_cnt = 0
    for cell in wrong_cells:
        if repaired_df.iloc[cell[0], cell[1]] == clean_df.iloc[cell[0], cell[1]]:
            rep_cnt = rep_cnt + 1
    rep_rec = rep_cnt/(len(wrong_cells)+1e-10)
    rep_f1 = 2*rep_pre*rep_rec/(rep_pre+rep_rec+1e-10)

    # if ONLYED:
    #     out_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Exp_result/holoclean/" + task_name[:-1] +"/onlyED_" + task_name + dirty_path[-25:-4] + ".txt"
    #     f = open(out_path, 'w')
    #     sys.stdout = f
    #     end_time = time.time()
    #     print("{pre}\n{rec}\n{f1}\n{time}".format(pre=det_pre, rec=det_rec, f1=det_f1, time=(end_time-start_time)))

    if not PERFECTED:
        out_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Exp_result/holoclean/" + task_name[:-1] + "/onlyED_" + task_name + check_string(dirty_path.split("/")[-1]) + ".txt"
        f = open(out_path, 'w')
        sys.stdout = f
        end_time = time.time()
        print("{pre}\n{rec}\n{f1}\n{time}".format(pre=det_pre, rec=det_rec, f1=det_f1, time=(end_time-start_time)))
        f.close()

        out_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Exp_result/holoclean/" + task_name[:-1] + "/all_compute_" + task_name + check_string(dirty_path.split("/")[-1]) + ".txt"
        f = open(out_path, 'w')
        sys.stdout = f
        right2wrong = 0
        right2right = 0
        wrong2right = 0
        wrong2wrong = 0
        clean_df = clean_df.applymap(str.lower)
        dirty_df = dirty_df.applymap(str.lower)
        repaired_df = repaired_df.applymap(str.lower)

        det_cells.append((det_df.iloc[i, 0], list(dirty_df).index(det_df.iloc[i, 1])))
        rep_cells = []
        repair_right_cells = []
        for tid, attr_vals in infer_val_df.items():
            for attr, val in attr_vals.items():
                cell = (tid, list(dirty_df.columns).index(attr))
                rep_cells.append(cell)
                if repaired_df.iloc[cell[0], cell[1]] == clean_df.iloc[cell[0], cell[1]]:
                    repair_right_cells.append(cell)

        clean_in_cands = []
        clean_in_cands_repair_right = []
        for i in range(len(domain)):
            cell = (domain.iloc[i]['_tid_'], list(dirty_df.columns).index(domain.iloc[i]['attribute']))
            cands = domain.iloc[i]['domain'].split('|||')
            if clean_df.iloc[cell[0], cell[1]] in cands:
                clean_in_cands.append(cell)
                if repaired_df.iloc[cell[0], cell[1]] == clean_df.iloc[cell[0], cell[1]]:
                    clean_in_cands_repair_right.append(cell)

        # for cell in clean_in_cands:
        #     if cell not in rep_cells:
        #         clean_in_cands.remove(cell)
        clean_in_cands =  [cell for cell in clean_in_cands if cell in rep_cells]
        # for cell in clean_in_cands_repair_right:
        #     if cell not in rep_cells:
        #         clean_in_cands_repair_right.remove(cell)
        clean_in_cands_repair_right = [cell for cell in clean_in_cands_repair_right if cell in rep_cells]

        rep_total = len(rep_cells)
        wrong_cell_num = len(wrong_cells)
        for cell in repair_right_cells:
            if cell in wrong_cells:
                wrong2right += 1
            else:
                right2right += 1

        rec_right = 0
        for cell in wrong_cells:
            if repaired_df.iloc[cell[0], cell[1]] == clean_df.iloc[cell[0], cell[1]]:
                rec_right += 1
        print("rep_right:"+str(len(repair_right_cells)))
        print("rec_right:"+str(rec_right))
        print("wrong_cells:"+str(wrong_cell_num))
        print("prec:"+str(len(repair_right_cells)/len(rep_cells)))
        print("rec:"+str(float(rec_right)/len(wrong_cells)))
        print("wrong2right:"+str(wrong2right))
        print("right2right:"+str(right2right))
        repair_wrong_cells = [i for i in rep_cells if i not in repair_right_cells]
        for cell in repair_wrong_cells:
            if cell in wrong_cells:
                wrong2wrong += 1
            else:
                right2wrong += 1
        print("wrong2wrong:"+str(wrong2wrong))
        print("right2wrong:"+str(right2wrong))
        print("proportion of clean value in candidates:"+str(len(clean_in_cands)/(rep_total+1e-8)))
        print("proportion of clean value in candidates and selected correctly:"+str(len(clean_in_cands_repair_right)/(len(clean_in_cands)+1e-8)))
        f.close()

        out_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Exp_result/holoclean/" + task_name[:-1] + "/oriED+EC_" + task_name + check_string(dirty_path.split("/")[-1]) + ".txt"
        res_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Repaired_res/holoclean/" + task_name[:-1] + "/repaired_" + task_name + check_string(dirty_path.split("/")[-1]) + ".csv"
        repaired_df.to_csv(res_path, index=False, columns=list(repaired_df.columns))
        f = open(out_path, 'w')
        sys.stdout = f
        rep_pre = len(repair_right_cells)/len(rep_cells)
        rep_rec = float(rec_right)/len(wrong_cells)
        rep_f1 = 2*rep_pre*rep_rec/(rep_pre+rep_rec+1e-10)
        # end_time = time.time()
        print("{pre}\n{rec}\n{f1}\n{time}".format(pre=rep_pre, rec=rep_rec, f1=rep_f1, time=(end_time-start_time)))
        f.close()

    else:
        out_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Exp_result/holoclean/" + task_name[:-1] + "/perfectED+EC_" + task_name + check_string(dirty_path.split("/")[-1]) + ".txt"
        res_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Repaired_res/holoclean/" + task_name[:-1] + "/perfect_repaired_" + task_name + check_string(dirty_path.split("/")[-1]) + ".csv"
        repaired_df.to_csv(res_path, index=False, columns=list(repaired_df.columns))
        f = open(out_path, 'w')
        sys.stdout = f
        end_time = time.time()
        print("{pre}\n{rec}\n{f1}\n{time}".format(pre=rep_pre, rec=rep_rec, f1=rep_f1, time=(end_time-start_time)))
