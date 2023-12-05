"""
This class defines the main experiment routines
"""
# import torch
import sys
import re
import time
import pandas as pd
import numpy as np
import argparse
sys.path.append('./BoostClean/')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from activedetect.reporting.CSVLogging import CSVLogging
from activedetect.loaders.csv_loader import CSVLoader
from activedetect.learning.baselines import *
from activedetect.learning.BoostClean import BoostClean
from activedetect.error_detectors.QuantitativeErrorModule import QuantitativeErrorModule
from activedetect.error_detectors.PuncErrorModule import PuncErrorModule
# from activedetect.learning.GAN_repair import gain
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from activedetect.learning.networks_gan_repair import DiscriminatorNet, GeneratorNet, ClassifierNet, Discriminator_loss, Generator_loss, Classifier_loss
# from torch.autograd import Variable
import datetime
from sklearn import neural_network
from sklearn.ensemble import RandomForestClassifier
from activedetect.learning.utils import normalization, renormalization, rounding

def check_string(string: str):
    if re.search(r"-inner_error-", string):
        return "-inner_error-" + string[-6:-4]
    elif re.search(r"-outer_error-", string):
        return "-outer_error-" + string[-6:-4]
    elif re.search(r"-inner_outer_error-", string):
        return "-inner_outer_error-" + string[-6:-4]
    elif re.search(r"-dirty-original_error-", string):
        return "-original_error-" + string[-9:-4]

class Experiment(object):

    def __init__(self,
                 features,
                 labels,
                 model,
                 experiment_name, 
                 wrong_cells = [],
                 PERFECTED = 0):
        """
        * features a list of lists
        * a list of binary attributes 1/0
        """

        self.features = features
        self.labels = labels
        logger = CSVLogging(experiment_name+".log")
        self.logger = logger
        self.model = model
        self.wrong_cells = wrong_cells
        self.PERFECTED = PERFECTED


    def runAllAccuracy(self):
        pass
        q_detect = QuantitativeErrorModule
        punc_detect = PuncErrorModule
        config = [{'thresh': 10}, {}]

        start = datetime.datetime.now()

        b = BoostClean(modules=[q_detect, punc_detect],
               config=config,
               base_model=self.model,
               features=self.features,
               labels=self.labels,
               logging=self.logger, 
               wrong_cells=wrong_cells,
               perfected=self.PERFECTED)
        _, repair_features, sel_clf = b.run(j=5)

        self.logger.logResult(["time_boostclean", str(datetime.datetime.now()-start)])
        return _, repair_features, sel_clf

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
    ONLYED = args.onlyed
    PERFECTED = args.perfected

    # dirty_path = "./data_with_rules/hospital/noise/hospital-outer_error-90.csv"
    # clean_path = "./data_with_rules/hospital/clean.csv"
    # task_name = "hospital1"
    # ONLYED = 0
    # PERFECTED = 0

    start_time = time.time()

    # c = CSVLoader()
    # dirty_data = c.loadFile(dirty_path)
    # clean_data = c.loadFile(clean_path)

    # dirty_data = dirty_data[1:]
    # clean_data = clean_data[1:]
    dirty_data = pd.read_csv(dirty_path).astype(str)
    dirty_data.fillna('nan', inplace=True)
    dirty_data = dirty_data.values.tolist()
    clean_data = pd.read_csv(clean_path).astype(str)
    clean_data.fillna('nan', inplace=True)
    columns = list(clean_data.columns)
    clean_data = clean_data.values.tolist()

    wrong_cells = []
    for i in range(len(dirty_data)):
        for j in range(len(dirty_data[0])):
            if clean_data[i][j] != dirty_data[i][j]:
                wrong_cells.append((i,j))

    # all but the last column are features
    features = [l[0:-1] for l in dirty_data]

    labels = [l[-1] for l in dirty_data]
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    # run the experiment, results are stored in uscensus.log
    # features, label, xsklearn model, name
    e = Experiment(features, labels, RandomForestClassifier(), task_name, wrong_cells, PERFECTED)
    _, rep_result, sel_clf = e.runAllAccuracy()
    rep_cells = sel_clf.error_cells
    # for i in range(len(rep_result)):
    #     for j in range(len(rep_result[0])):
    #         if rep_result[i][j] != dirty_data[i][j]:
    #             rep_cells.append((i, j))

    for i in range(len(sel_clf.training_labels_copy)):
        rep_result[i][-1] = le.inverse_transform([rep_result[i][-1]])[0]
    for i in range(len(sel_clf.training_labels_copy), len(rep_result)):
        rep_result[i].append(dirty_data[i][-1])
    rep_cells = list(set(rep_cells))
    wrong_cells = list(set(wrong_cells))
    det_right = 0

    

    if True:
        if not PERFECTED:
            out_path = "./Exp_result/boostclean/" + task_name[:-1] + "/onlyED_" + task_name + check_string(dirty_path) + ".txt"
            f = open(out_path, 'w')
            sys.stdout = f
            end_time = time.time()
            for cell in rep_cells:
                if cell in wrong_cells:
                    det_right = det_right + 1
            pre = det_right / (len(rep_cells)+1e-10)
            rec = det_right / (len(wrong_cells)+1e-10)
            f1 = 2*pre*rec/(pre+rec+1e-10)
            print("{pre}\n{rec}\n{f1}\n{time}".format(pre=pre, rec=rec, f1=f1, time=(end_time-start_time)))
            f.close()

            out_path = "./Exp_result/boostclean/" + task_name[:-1] + "/oriED+EC_" + task_name + check_string(dirty_path) + ".txt"
            res_path = "./Repaired_res/boostclean/" + task_name[:-1] + "/repaired_" + task_name + check_string(dirty_path) + ".csv"
            # for res in rep_result:
            #     if len(res) == 11:
            #         res.remove(res[-3])
            rep_csv = pd.DataFrame(np.array(rep_result).reshape(-1, len(columns)), columns=columns)
            rep_csv.to_csv(res_path, index=False, columns=list(rep_csv.columns)[0:])
            f = open(out_path, 'w')
            sys.stdout = f
            end_time = time.time()
            rep_right = 0
            rep_total = len(rep_cells)
            # wrong_cells = len(wrong_cells)
            rec_right = 0
            for cell in rep_cells:
                if rep_result[cell[0]][cell[1]] == clean_data[cell[0]][cell[1]]:
                    rep_right += 1
            for cell in wrong_cells:
                if cell[0] >= len(rep_result):
                    continue
                if rep_result[cell[0]][cell[1]] == clean_data[cell[0]][cell[1]]:
                    rec_right += 1
            pre = rep_right / (rep_total+1e-10)
            rec = rec_right / (len(wrong_cells)+1e-10)
            f1 = 2*pre*rec / (rec+pre+1e-10)
            print("{pre}\n{rec}\n{f1}\n{time}".format(pre=pre, rec=rec, f1=f1, time=(end_time-start_time)))
            f.close()
        else:
            out_path = "./Exp_result/boostclean/" + task_name[:-1] + "/prefectED+EC_" + task_name + check_string(dirty_path) + ".txt"
            res_path = "./Repaired_res/boostclean/" + task_name[:-1] + "/perfect_repaired_" + task_name + check_string(dirty_path) + ".csv"
            for res in rep_result:
                if len(res) == 11:
                    res.remove(res[-3])
            rep_csv = pd.DataFrame(np.array(rep_result), columns=columns)
            rep_csv.to_csv(res_path, index=False, columns=list(rep_csv.columns)[0:])
            f = open(out_path, 'w')
            sys.stdout = f
            end_time = time.time()
            rep_right = 0
            rep_total = len(rep_cells)
            # wrong_cells = len(wrong_cells)
            rec_right = 0
            rep_t = 0
            for cell in wrong_cells:
                if cell in rep_cells:
                    rep_t += 1
                    if cell[0] >= len(rep_result):
                        continue
                    if rep_result[cell[0]][cell[1]] == clean_data[cell[0]][cell[1]]:
                        rec_right += 1
            pre = rec_right / (rep_t+1e-10)
            rec = rec_right / (len(wrong_cells)+1e-10)
            f1 = 2*pre*rec / (rec+pre+1e-10)
            print("{pre}\n{rec}\n{f1}\n{time}".format(pre=pre, rec=rec, f1=f1, time=(end_time-start_time)))
            f.close()
     