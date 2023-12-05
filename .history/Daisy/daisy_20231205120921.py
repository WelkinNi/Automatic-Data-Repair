import copy
import daisy
import pandas as pd
import time
import argparse
import sys


if __name__ == '__main__':
    # dataset = daisy.dataset()
    # df = dataset.df
    # dict = dataset.dict

    # select1 = daisy.Select(select_attribute=['index','provider_number','name'],from_table='hospital',where_caluse=['index','<',10])
    # fd = daisy.FD(lhs='name',rhs='address_1')

#-------------4.1-----------------
    # dataset = daisy.dataset()
    # df = dataset.df
    # dict = dataset.dict
    # select1 = daisy.Select(select_attribute=['Zip'], from_table='city',
    #                        where_caluse=['City', '==', 'Los Angeles'])
    # fd = daisy.FD(lhs='Zip', rhs='City')
    # result = daisy.do_select(dataset, select1)
    # cleansigma1 = daisy.CleanSigma(dataset, result, fd=fd)  # 传入dataset对象、result字典，fd对象

# -------------4.2-----------------
#     dataset = daisy.dataset()
#     df = dataset.df
#     dict = dataset.dict
    # select1 = daisy.Select(select_attribute=['name','salary'],from_table='salary',
    #                        where_caluse=['salary','>=',1000])
    # dc = daisy.DCRule('t1&t2&EQ(t1.name,t2.name)',['name'])


#--------------4.3----------------
    #---------------------------Multiple FD--------------------
    # dataset = daisy.dataset()
    # df = dataset.df
    # dict = dataset.dict
    # select1 = daisy.Select(select_attribute=['Zip','City','State'], from_table='city_state',
    #                        where_caluse=['City', '==', 'Los Angeles'])
    # fd1 = daisy.FD(lhs='Zip', rhs='State')
    # fd2 = daisy.FD(lhs='City',rhs='State')
    # result = daisy.do_select(dataset, select1)
    # cleansigma1 = daisy.CleanSigma(dataset, result, fd=fd1)  # 传入dataset对象、result字典，fd1对象
    # cleansigma2 = daisy.CleanSigma(dataset, result, fd=fd2)  # 传入dataset对象、result字典，fd1对象
    # fd_list = [fd1,fd2]
    # cleansigma_list = [cleansigma1,cleansigma2]
    # multicleansigma = daisy.MultiCleanSigma(dataset,cleansigma_list,fd_list=fd_list)

# ---------------------------Multiple DC--------------------
#     dataset = daisy.dataset()
#     df = dataset.df
#     dict = dataset.dict
#     select1 = daisy.Select(select_attribute=['name','salary','tax','age'], from_table='salary_age',
#                            where_caluse=['salary', '<=', 2500])
#     dc1 = daisy.DCRule('t1&t2&LT(t1.salary,t2.salary)&GT(t1.tax,t2.tax)',['salary','tax'])
#     dc2 = daisy.DCRule('t1&t2&LT(t1.salary,t2.salary)&GT(t1.age,t2.age)',['salary','age'])
#     result = daisy.do_select(dataset, select1)
#     cleansigma1 = daisy.CleanSigma(dataset, result, dc = dc1)
#     cleansigma2 = daisy.CleanSigma(dataset, result, dc = dc2)
#     dc_list = [dc1,dc2]
#     cleansigma_list = [cleansigma1,cleansigma2]
#     multicleansigma = daisy.MultiCleanSigma(dataset,cleansigma_list,dc_list=dc_list)

#--------------4.4----------------
    # dataset1 = daisy.dataset()
    # df1 = dataset1.df
    # dict1 = dataset1.dict
    # dataset2 = daisy.dataset()
    # df2 = dataset2.df
    # dict2 = dataset2.dict
    # select = daisy.Select(select_attribute=['Zip'],from_table='city_join',
    #                       where_caluse=['City','==','Los Angeles'])
    # fd1 = daisy.FD(lhs='Zip',rhs='City')
    # fd2 = daisy.FD(lhs='Phone',rhs='Zip')
    # join = daisy.Join(table1='city_join',table2='employee_join',join_key='Zip',project_attri=['Zip','Name'],
    #                   select=select,fd1=fd1,fd2=fd2)
    # result = daisy.do_join(dataset1,dataset2,join)
    # cleanjoin = daisy.CleanJoin(result,dataset1,dataset2,join)
    

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

    # dirty_path = "./data with dc_rules/hospital/noise/hospital-inner_error-10.csv"
    # clean_path = "./data with dc_rules/hospital/clean.csv"
    # rule_path = "./data with dc_rules/hospital/dc_rules_holoclean.txt"
    # PERFECTED = 0

    start_time = time.time()
    dataset = daisy.dataset(dirty_path)
    df = dataset.df
    dict = dataset.dict
    # dict = daisy_hospital.basic.hospital_basic(dict)
    select = daisy.Select(select_attribute=list(df.columns),
                                                from_table='dirty',
                                                where_caluse=['index','>=',1])
    clean_csv = pd.read_csv(clean_path)
    f = open(rule_path, 'r')
    dc_list = []
    cleansigma_list = []
    for line in f.readlines():
        dc = daisy.DCRule(line, list(clean_csv.columns))
        result = daisy.do_select(dataset,select)
        cleansigma_temp = daisy.CleanSigma(dataset,result,dc=dc)
        dc_list.append(dc)
        cleansigma_list.append(cleansigma_temp)
    multicleansigma = daisy.MultiCleanSigma(dataset,cleansigma_list,dc_list=dc_list)
    rep_csv = multicleansigma.dataset.df.iloc[:, 1:]
    dirty_csv = dataset.df.iloc[:, 1:]
    rep_cells = []
    wrong_cells_list = []
    for i in range(len(dirty_csv)):
        for j in range(len(dirty_csv.columns)):
            if clean_csv.iloc[i, j] != dirty_csv.iloc[i, j]:
                wrong_cells_list.append((i, j))
    for i in range(len(dirty_csv)):
        for j in range(len(dirty_csv.columns)):
            if rep_csv.iloc[i, j] != dirty_csv.iloc[i, j]:
                rep_cells.append((i, j))
    if not PERFECTED:
        det_right = 0
        out_path = "./Exp_result/daisy/" + task_name[:-1] + "/onlyED_" + task_name + dirty_path[-25:-4] + ".txt"
        f = open(out_path, 'w')
        sys.stdout = f
        end_time = time.time()
        for cell in rep_cells:
            if cell in wrong_cells_list:
                det_right = det_right + 1
        pre = det_right / (len(rep_cells)+1e-10)
        rec = det_right / (len(wrong_cells_list)+1e-10)
        f1 = 2*pre*rec/(pre+rec+1e-10)
        print("{pre}\n{rec}\n{f1}\n{time}".format(pre=pre, rec=rec, f1=f1, time=(end_time-start_time)))
        f.close()

        out_path = "./Exp_result/daisy/" + task_name[:-1] + "/oriED+EC_" + task_name + dirty_path[-25:-4] + ".txt"
        res_path = "./Repaired_res/daisy/" + task_name[:-1] + "/repaired_" + task_name + dirty_path[-25:-4] + ".csv"
        rep_csv.to_csv(res_path, index=False)
        f = open(out_path, 'w')
        sys.stdout = f
        end_time = time.time()
        rep_right = 0
        rep_total = len(rep_cells)
        wrong_cell_num = len(wrong_cells_list)
        rec_right = 0
        for cell in rep_cells:
            if rep_csv.iloc[cell[0], cell[1]] == clean_csv.iloc[cell[0], cell[1]]:
                rep_right += 1
        for cell in wrong_cells_list:
            if rep_csv.iloc[cell[0], cell[1]] == clean_csv.iloc[cell[0], cell[1]]:
                rec_right += 1
        pre = rep_right / (rep_total+1e-10)
        rec = rec_right / (wrong_cell_num+1e-10)
        f1 = 2*pre*rec / (rec+pre+1e-10)
        print("{pre}\n{rec}\n{f1}\n{time}".format(pre=pre, rec=rec, f1=f1, time=(end_time-start_time)))
        f.close()


