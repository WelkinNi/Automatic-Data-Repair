import copy
import daisy
import daisy_beer
import daisy_hospital
import daisy_flight
import pandas as pd
import time
import argparse
import sys

#  E:\Graduation Project\Daisy\datasets\hospital\dirty.csv
#  E:\Graduation Project\Daisy\datasets\hospital\city.csv
#  E:\Graduation Project\Daisy\datasets\hospital\salary.csv

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


#------------beer数据集处理,基本处理与单属性DC处理--------------
    # dataset = daisy.dataset()
    # df = dataset.df
    # dict = dataset.dict
    # dict = daisy_beer.basic.beer_basic(dict)
    # # dict_original = copy.deepcopy(dict)
    # select1 = daisy.Select(select_attribute=['index', 'id','beer_name','style',
    #                                          'ounces','abv','ibu','brewery_id','brewery_name','city','state'],
    #                                             from_table='dirty',
    #                                               where_caluse=['index','>=',1])
    # #∀t0∈clean.csv,t1∈clean.csv:¬[t0.beer-name=t1.beer-name]
    # dc = daisy.DCRule('t1&t2&EQ(t1.beer_name,t2.beer_name)', ['beer_name'])
    # result = daisy.do_select(dataset,select1)
    # cleansigma1 = daisy.CleanSigma(dataset,result,dc=dc)#传入dataset对象、result字典，fd对象
    #
    # #∀t0∈clean.csv, t1∈clean.csv:¬[t0.id = t1.id]
    # dc1 = daisy.DCRule('t1&t2&EQ(t1.id,t2.id)',['id'])
    # dataset.dict = cleansigma1.relax_full
    # result1 = daisy.do_select(dataset,select1)
    # cleansigma2 = daisy.CleanSigma(dataset,result1,dc=dc1)
    #
    # dict_original = copy.deepcopy(cleansigma2.relax_full)
    #
    # #∀t0∈clean.csv,t1∈clean.csv:¬[t0.style=t1.style∧t0.ibu=t1.ibu]
    # dc = daisy.DCRule('t1&t2&EQ(t1.style,t2.style)&EQ(t1.ibu,t2.ibu)',['ibu','style'])
    # dataset.dict = copy.deepcopy(dict_original)
    # result = daisy.do_select(dataset, select1)
    # cleansigma3 = daisy.CleanSigma(dataset, result, dc=dc)
    # pass

#------------hospital数据集处理-----------------------
    # start_time = time.time()
    # dataset = daisy.dataset("/data/nw/DC_ED/References_inner_and_outer/Daisy/daisy_hospital/dirty.csv")
    # df = dataset.df
    # dict = dataset.dict
    # # dict = daisy_hospital.basic.hospital_basic(dict)
    
    # select = daisy.Select(select_attribute=list(df.columns),
    #                                             from_table='dirty',
    #                                             where_caluse=['index','>=',1])
    # # ¬[t0.ZipCode=t1.ZipCode∧t0.State≠t1.State]
    # dc1 = daisy.DCRule('t1&t2&EQ(t1.zip,t2.zip)&IQ(t1.state,t2.state)', ['zip','state'])
    # result = daisy.do_select(dataset,select)
    # cleansigma1 = daisy.CleanSigma(dataset,result,dc=dc1)
    
    # # ¬[t0.EmergencyService≠t1.EmergencyService∧t0.HospitalOwner=t1.HospitalOwner∧t0.Score=t1.Score]
    # dc2 = daisy.DCRule('t1&t2&EQ(t1.score,t2.score)&EQ(t1.owner,t2.owner)&IQ(t1.emergency_service,t2.emergency_service',
    #                   ['score','owner','emergency_service'])
    # result = daisy.do_select(dataset, select)
    # cleansigma2 = daisy.CleanSigma(dataset, result, dc=dc2)
    
    # #¬[t0.PhoneNumber≥t1.PhoneNumber∧t0.HospitalOwner≠t1.HospitalOwner∧t0.CountyName=t1.CountyName]
    # dc3 = daisy.DCRule('t1&t2&EQ(t1.county,t2.county)&GTE(t1.phone,t2.phone)&IQ(t1.owner,t2.owner)',
    #                    ['county','phone','owner'])
    # result = daisy.do_select(dataset, select)
    # cleansigma3 = daisy.CleanSigma(dataset, result, dc=dc3)
    
    # #¬[t0.ZipCode<t1.ZipCode∧t0.ProviderNumber≤t1.ProviderNumber∧
    # # t0.EmergencyService≠t1.EmergencyService∧t0.Score=t1.Score]
    # dc4 = daisy.DCRule('t1&t2&EQ(t1.score,t2.score)&IQ(t1.emergency_service,t2.emergency_service)'
    #                    '&LT(t1.zip,t2.zip)&LTE(t1.provider_number,t2.provider_number)',
    #                    ['score','emergency_service','zip','provider_number'])
    # result = daisy.do_select(dataset, select)
    # cleansigma4 = daisy.CleanSigma(dataset, result, dc=dc4)
    
    # #¬[t0.HospitalOwner=t1.HospitalOwner∧t0.PhoneNumber<t1.PhoneNumber∧
    # # t0.ProviderNumber≥t1.ProviderNumber∧t0.ZipCode≥t1.ZipCode]
    # dc5 = daisy.DCRule('t1&t2&EQ(t1.owner,t2.owner)&LT(t1.phone,t2.phone)'
    #                    '&GTE(t1.provider_number,t2.provider_number)&GTE(t1.zip,t2.zip)',
    #                    ['owner','phone','provider_number','zip'])
    # result = daisy.do_select(dataset, select)
    # cleansigma5 = daisy.CleanSigma(dataset, result, dc=dc5)
    
    # #¬[t0.ZipCode<t1.ZipCode∧t0.Condition=t1.Condition∧t0.PhoneNumber≥t1.PhoneNumber∧
    # # t0.EmergencyService≠t1.EmergencyService]e]
    # dc6 = daisy.DCRule('t1&t2&EQ(t1.condition,t2.condition)&LT(t1.zip,t2.zip)'
    #                    '&GTE(t1.phone,t2.phone)&IQ(t1.emergency_service,t2.emergency_service)',
    #                    ['condition', 'zip', 'phone', 'emergency_service'])
    # result = daisy.do_select(dataset, select)
    # cleansigma6 = daisy.CleanSigma(dataset, result, dc=dc6)
    
    # #¬[t0.index≥t1.index∧t0.ProviderNumber<t1.ProviderNumber∧t0.HospitalOwner=t1.HospitalOwner∧
    # # t0.ZipCode≤t1.ZipCode]
    # dc7 = daisy.DCRule('t1&t2&EQ(t1.owner,t2.owner)&LTE(t1.zip,t2.zip)'
    #                    '&LT(t1.provider_number,t2.provider_number)&GTE(t1.index,t2.index)',
    #                    ['owner', 'zip', 'provider_number', 'index'])
    # result = daisy.do_select(dataset, select)
    # cleansigma7 = daisy.CleanSigma(dataset, result, dc=dc7)
    
    # #¬[t0.HospitalOwner=t1.HospitalOwner∧t0.ProviderNumber≥t1.ProviderNumber∧t0.ZipCode≥t1.ZipCode∧
    # # t0.Address1≠t1.Address1∧t0.index≤t1.index]
    # dc8 = daisy.DCRule('t1&t2&EQ(t1.owner,t2.owner)&GTE(t1.provider_number,t2.provider_number)'
    #                    '&GTE(t1.zip,t2.zip)&IQ(t1.address_1,t2.address_1)&LTE(t1.index,t2.index)',
    #                    ['owner', 'provider_number', 'zip', 'address_1','index'])
    # result = daisy.do_select(dataset, select)
    # cleansigma8 = daisy.CleanSigma(dataset, result, dc=dc8)
    
    # dc_list = [dc1,dc2,dc3,dc4,dc5,dc6,dc7,dc8]
    # cleansigma_list = [cleansigma1,cleansigma2,cleansigma3,cleansigma4,cleansigma5,cleansigma6,cleansigma7,cleansigma8]
    # multicleansigma = daisy.MultiCleanSigma(dataset,cleansigma_list,dc_list=dc_list)
    # print(multicleansigma.relax_result)
    # end_time = time.time()
    # print(str(end_time-start_time) + "s")
    

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

    # dirty_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/data with dc_rules/hospital/noise/hospital-inner_error-10.csv"
    # clean_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/data with dc_rules/hospital/clean.csv"
    # rule_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/data with dc_rules/hospital/dc_rules_holoclean.txt"
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
        out_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Exp_result/daisy/" + task_name[:-1] + "/onlyED_" + task_name + dirty_path[-25:-4] + ".txt"
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

        out_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Exp_result/daisy/" + task_name[:-1] + "/oriED+EC_" + task_name + dirty_path[-25:-4] + ".txt"
        res_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Repaired_res/daisy/" + task_name[:-1] + "/repaired_" + task_name + dirty_path[-25:-4] + ".csv"
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


# ------------flight数据集处理-----------------------
    # dataset = daisy.dataset('/data/nw/DC_ED/References_inner_and_outer/Daisy/daisy_flight/dirty.csv')
    # df = dataset.df
    # dict = dataset.dict
    # # dict = daisy_flight.basic.flight_basic(dict)

    # select = daisy.Select(select_attribute=['tuple_id','src','flight','sched_dep_time','act_dep_time',
    #                                         'sched_arr_time','act_arr_time'],
    #                                             from_table='dirty',
    #                                               where_caluse=['tuple_id','>=',1])

    # # ¬[t0.src=t1.src∧t0.act_dep_time=t1.act_dep_time]
    # dc1 = daisy.DCRule('t1&t2&EQ(t1.act_dep_time,t2.act_dep_time)&EQ(t1.src,t2.src)', ['act_dep_time','src'])
    # result = daisy.do_select(dataset,select)
    # cleansigma1 = daisy.CleanSigma(dataset,result,dc=dc1)

    # #¬[t0.src = t1.src∧t0.act_arr_time = t1.act_arr_time]
    # dc2 = daisy.DCRule('t1&t2&EQ(t1.act_arr_time,t2.act_arr_time)&EQ(t1.src,t2.src)', ['act_arr_time', 'src'])
    # result = daisy.do_select(dataset, select)
    # cleansigma2 = daisy.CleanSigma(dataset, result, dc=dc2)

    # #¬[t0.sched_dep_time=t1.sched_dep_time∧t0.flight≠t1.flight]
    # dc3 = daisy.DCRule('t1&t2&EQ(t1.sched_dep_time,t2.sched_dep_time)&IQ(t1.flight,t2.flight)', ['sched_dep_time', 'flight'])
    # result = daisy.do_select(dataset, select)
    # cleansigma3 = daisy.CleanSigma(dataset, result, dc=dc3)

    # #¬[t0.act_arr_time≠t1.act_arr_time∧t0.act_dep_time=t1.act_dep_time]
    # dc4 = daisy.DCRule('t1&t2&EQ(t1.act_dep_time,t2.act_dep_time)&IQ(t1.act_arr_time,t2.act_arr_time)',['act_dep_time', 'act_arr_time'])
    # result = daisy.do_select(dataset, select)
    # cleansigma4 = daisy.CleanSigma(dataset, result, dc=dc4)

    # #¬[t0.act_dep_time≠t1.act_dep_time∧t0.act_arr_time=t1.act_arr_time]
    # dc5 = daisy.DCRule('t1&t2&EQ(t1.act_arr_time,t2.act_arr_time)&IQ(t1.act_dep_time,t2.act_dep_time)',['act_arr_time', 'act_dep_time'])
    # result = daisy.do_select(dataset, select)
    # cleansigma5 = daisy.CleanSigma(dataset, result, dc=dc5)

    # #¬[t0.act_arr_time≠t1.act_arr_time∧t0.sched_arr_time=t1.sched_arr_time]
    # dc6 = daisy.DCRule('t1&t2&EQ(t1.sched_arr_time,t2.sched_arr_time)&IQ(t1.act_arr_time,t2.act_arr_time)',['sched_arr_time', 'act_arr_time'])
    # result = daisy.do_select(dataset, select)
    # cleansigma6 = daisy.CleanSigma(dataset, result, dc=dc6)

    # #¬[t0.sched_arr_time≠t1.sched_arr_time∧t0.act_arr_time=t1.act_arr_time]
    # dc7 = daisy.DCRule('t1&t2&EQ(t1.act_arr_time,t2.act_arr_time)&IQ(t1.sched_arr_time,t2.sched_arr_time)',['act_arr_time', 'sched_arr_time'])
    # result = daisy.do_select(dataset, select)
    # cleansigma7 = daisy.CleanSigma(dataset, result, dc=dc7)

    # dc_list = [dc1,dc2,dc3,dc4,dc5,dc6,dc7]
    # cleansigma_list = [cleansigma1,cleansigma2,cleansigma3,cleansigma4,cleansigma5,cleansigma6,cleansigma7]
    # multicleansigma = daisy.MultiCleanSigma(dataset,cleansigma_list,dc_list=dc_list)
    # print(multicleansigma.relax_result)
    # pass