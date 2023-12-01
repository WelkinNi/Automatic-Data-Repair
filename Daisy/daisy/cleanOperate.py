import copy
import math
import pandas as pd
import string

import daisy

class CleanSigma:
    dataset = {}
    answer = {}

    def __init__(self,dataset,answer,fd = None,dc = None):
        self.dataset = dataset
        self.answer = answer
        self.fd = fd
        self.dc = dc
        if fd != None:
            self.rela_ans_tup,self.rela_extr_tup = self.result_relaxion_fd()
        elif dc != None:
            self.result_relaxion_dc()

    # -------------对FD的relax----------------
    def result_relaxion_fd(self):
        total_extra = {}
        unvisited = self.difference(self.dataset, self.answer) #返回一个字典，answer以外的所有原tuple
        answer_full_tuple = {}
        unvisited_full_tuple = {}
        for i in range(len(self.answer)):
            answer_full_tuple[i] = self.dataset.dict[self.answer[i]['original-index']]
        for i in range(len(unvisited)):
            unvisited_full_tuple[i] = self.dataset.dict[unvisited[i].name]
        A_lhs = set()
        A_rhs = set()
        for key in answer_full_tuple.keys():
            A_lhs.add(answer_full_tuple[key][self.fd.lhs])
            A_rhs.add(answer_full_tuple[key][self.fd.rhs])
        extra = {} #存储对应映射下的额外相关元组
        extra_full = {} #存储相关元组的所有属性均存在情况
        i = 0
        for key in unvisited_full_tuple.keys():
            if unvisited_full_tuple[key][self.fd.lhs] in A_lhs:
                extra[i] = unvisited[key]
                extra_full[i] = unvisited_full_tuple[key]
                i = i + 1
            elif unvisited_full_tuple[key][self.fd.rhs] in A_rhs:
                extra[i] = unvisited[key]
                extra_full[i] = unvisited_full_tuple[key]
                i = i + 1
        #-------------计算概率----------------
        relax_answer_tuple = {}
        relax_extra_tuple = {}
        for key in answer_full_tuple.keys():
            lhs_prob_res,rhs_prob_res = self.cal_probablistic(answer_full_tuple[key], self.dataset.dict)
            temp2 = copy.deepcopy(answer_full_tuple[key])
            temp2[self.fd.lhs] = lhs_prob_res
            temp3 = copy.deepcopy(answer_full_tuple[key])
            temp3[self.fd.rhs] = rhs_prob_res
            temp1 = {}
            temp1['candidate1'] = temp2
            temp1['candidate2'] = temp3
            relax_answer_tuple[key] = temp1
        for key in extra_full.keys():
            lhs_prob_res, rhs_prob_res = self.cal_probablistic(extra_full[key], self.dataset.dict)
            temp2 = copy.deepcopy(extra_full[key])
            temp2[self.fd.lhs] = lhs_prob_res
            temp3 = copy.deepcopy(extra_full[key])
            temp3[self.fd.rhs] = rhs_prob_res
            temp1 = {}
            temp1['candidate1'] = temp2
            temp1['candidate2'] = temp3
            relax_extra_tuple[key] = temp1

        return relax_answer_tuple,relax_extra_tuple

    # -------------对DC的relax----------------
    def result_relaxion_dc(self):
        unvisited = self.difference(self.dataset, self.answer)  # 返回一个字典，answer以外的所有原tuple
        answer_full_tuple = {}
        unvisited_full_tuple = {}
        for i in range(len(self.answer)):
            answer_full_tuple[i] = self.dataset.dict[self.answer[i]['original-index']]
        for i in range(len(unvisited)):
            unvisited_full_tuple[i] = self.dataset.dict[unvisited[i].name]
        A_correlate = {}
        for i in range(len(self.dc.schema)):
            if self.dc.schema[i] in A_correlate.keys():
                pass
            else:
                value_set = set()
                for key in self.answer.keys():
                    value_set.add(self.answer[key][self.dc.schema[i]])
                A_correlate[self.dc.schema[i]] = value_set

        extra = {}  # 存储对应映射下的额外相关元组
        extra_full = {}  # 存储相关元组的所有属性均存在情况
        i = 0
        for key in unvisited_full_tuple.keys():
            for _key in A_correlate.keys():
                if unvisited_full_tuple[key][_key] in A_correlate[_key]:
                    extra[i] = unvisited[key]
                    extra_full[i] = unvisited_full_tuple[key]
                    i = i + 1
                    break
        #---------------------进行self-theta-join-------------------
        correlated_full,map_matrix,is_str = self.self_theta_join(answer_full_tuple,extra_full)
        #---------------------根据操作符对映射矩阵进行剪枝---------------
        self.prune_matrix(correlated_full,map_matrix,is_str)
        #---------------------如果DC涉及的属性超过一个------------------
        if len(self.dc.schema) > 1:
            #---------------------进行DC violation的detection,得到candidate-----------
            candidate_full,freq_dict = self.detect_DC_violation(map_matrix,correlated_full)
            #---------------------计算概率-----------------------------
            self.cal_prob(candidate_full,freq_dict)
        #--------------------如果该DC仅涉及一个元素，即不允许重复类型---------
        else:
            candidate_full = self.detectAndrepair_repetition(map_matrix,correlated_full)
        self.relax_full = candidate_full
        pass

    def difference(self,dataset,answer):
        project_list = list(answer[0].index)
        extra = copy.deepcopy(dataset)
        for i in range(len(answer)):
            del extra.dict[answer[i]['original-index']]
            extra.row = extra.row - 1
        select = daisy.Select(select_attribute=project_list,from_table= dataset.name,where_caluse=[])
        extra_dict = daisy.do_select(extra,select)
        return extra_dict


    def cal_probablistic(self,tuple,dataset):
        lhs_attri = self.fd.lhs
        rhs_attri = self.fd.rhs
        lhs_attri_num = {}
        rhs_attri_num = {}
        lhs_attri_total = 0
        rhs_attri_total = 0
        for key in dataset.keys():
            if dataset[key][rhs_attri] == tuple[rhs_attri] :
                if lhs_attri_num.__contains__(dataset[key][lhs_attri]):
                    lhs_attri_num[dataset[key][lhs_attri]] = lhs_attri_num[dataset[key][lhs_attri]]+1
                    lhs_attri_total = lhs_attri_total + 1
                else:
                    lhs_attri_num[dataset[key][lhs_attri]] = 1
                    lhs_attri_total = lhs_attri_total + 1
            if dataset[key][lhs_attri] == tuple[lhs_attri] :
                if rhs_attri_num.__contains__(dataset[key][rhs_attri]):
                    rhs_attri_num[dataset[key][rhs_attri]] = rhs_attri_num[dataset[key][rhs_attri]]+1
                    rhs_attri_total = rhs_attri_total + 1
                else:
                    rhs_attri_num[dataset[key][rhs_attri]] = 1
                    rhs_attri_total = rhs_attri_total + 1
        lhs_probablistic_result = {}
        rhs_probablistic_result = {}
        i = 0
        for key in lhs_attri_num:
            key_proba = lhs_attri_num[key] / lhs_attri_total
            lhs_probablistic_result[key] = key_proba
        for key in rhs_attri_num:
            key_proba = rhs_attri_num[key] / rhs_attri_total
            rhs_probablistic_result[key] = key_proba
        return lhs_probablistic_result,rhs_probablistic_result


    def self_theta_join(self,answer_full,extra_full):
        is_str = False
        corre_full = {}
        i = 0
        for key in answer_full.keys():
            corre_full[i] = answer_full[key]
            i = i + 1
        for key in extra_full.keys():
            corre_full[i] = extra_full[key]
            i = i + 1
        map_attri = self.dc.schema[0] #选取第一个属性作为map矩阵的指标
        # 如果比较的属性是str 那么matrix的横纵坐标按照对应index分割 不再按照属性值分割
        # 且将值一样（或不一样）的tuple pair放在一起
        if isinstance(map_attri,str):
            is_str = True
            min_value = 0
            max_value = len(corre_full)
            if len(corre_full) > 10:
                seperate = 10
                step_width = (max_value - min_value) / seperate
            else:
                seperate = len(corre_full)
                step_width = 1
            map_matrix = {}
            for i in range(seperate):
                for j in range(seperate):
                    map_matrix[(i, j)] = [(-1, -1)]
            for i in range(len(corre_full)):
                for j in range(len(corre_full)):
                    tuple = (i, j)
                    a = int(i / step_width)
                    b = int(j / step_width)
                    if (a == seperate):
                        a = a - 1
                    if (b == seperate):
                        b = b - 1
                    map_matrix[(a, b)].append(tuple)
            for i in range(seperate):
                for j in range(seperate):
                    del map_matrix[(i, j)][0]
            return corre_full, map_matrix,is_str
        #如果比较的属性是数字
        else:
            max_value = corre_full[0][map_attri]
            min_value = corre_full[0][map_attri]
            for key in corre_full.keys():
                if corre_full[key][map_attri] > max_value:
                    max_value = corre_full[key][map_attri]
                if corre_full[key][map_attri] < min_value:
                    min_value = corre_full[key][map_attri]
            if len(corre_full) > 10:
                seperate = 10
                step_width = (max_value-min_value)/seperate
            else:
                seperate = len(corre_full)
                step_width = (max_value-min_value)/seperate
            map_matrix = {}
            for i in range(seperate):
                for j in range(seperate):
                    map_matrix[(i,j)] = [(-1,-1)]
            for i in range(len(corre_full)):
                for j in range(len(corre_full)):
                    tuple = (i,j)
                    a = int((corre_full[i][map_attri]-min_value) / step_width)
                    b = int((corre_full[j][map_attri]-min_value) / step_width)
                    if(a == seperate):
                        a = a - 1
                    if(b == seperate):
                        b = b - 1
                    map_matrix[(a,b)].append(tuple)
            for i in range(seperate):
                for j in range(seperate):
                    del map_matrix[(i,j)][0]
            # map_matrix[(a,b)]代表
            # 行为[min_value+a*step_width，min_value+（a+1）*step_width),
            # 列为[min_value+b*step_width，min_value+（b+1）*step_width)的块
            return corre_full,map_matrix,is_str


    def prune_matrix(self,correlated_full,map_matrix,is_str):
        length = int(math.sqrt(len(map_matrix)))
        #将常数放在第一个predicte的后方，则判断出为这种情况则放弃剪枝
        if self.dc.predicates[0].property[1] == 'constant':
            return
        #如果进行θ-join的元素是数字
        if is_str == False:
            attri_operator = self.dc.predicates[0].opt
            if attri_operator == '=':
                for i in range(length):
                    for j in range(length):
                        if i != j:
                            map_matrix[(i,j)].clear()
            elif attri_operator == '<' or attri_operator == '<=':
                for i in range(length):
                    for j in range(length):
                        if i > j:
                            map_matrix[(i, j)].clear()
            elif attri_operator == '>' or attri_operator == '>=':
                for i in range(length):
                    for j in range(length):
                        if i < j:
                            map_matrix[(i, j)].clear()
        #如果进行θ-join的元素是str
        else:
            compare_attri = self.dc.schema[0]
            attri_operator = self.dc.predicates[0].opt
            if attri_operator == '=':
                for i in range(length):
                    for j in range(length):
                        list_tmp = map_matrix[(i,j)]
                        for k in range(len(list_tmp)):
                            if list_tmp[k][0] == list_tmp[k][1]:
                                list_tmp[k] = -1
                                continue
                            if correlated_full[list_tmp[k][0]][compare_attri] != correlated_full[list_tmp[k][1]][compare_attri]:
                                list_tmp[k] = -1
                        for l in range(len(list_tmp) - 1, -1,-1):  # 倒序循环，从最后一个元素循环到第一个元素。不能用正序循环，因为正序循环删除元素后，后续的列表的长度和元素下标同时也跟着变了，由于len(alist)是动态的。
                            if list_tmp[l] == -1:
                                list_tmp.pop(l)
                pass


    def detect_DC_violation(self,map_matrix,correlated_full):
        dc_attri_set = self.dc.schema
        matrix_length = int(math.sqrt(len(map_matrix)))
        candidate_full = {}
        fre_dict = {}
        for dc_it in range(len(self.dc.predicates)):
            for attri in self.dc.predicates[dc_it].components:
                fre_dict[attri] = {}
            # attri1 = self.dc.predicates[dc_it].components[0]
            # attri2 = self.dc.predicates[dc_it].components[1]
            # fre_dict[attri1] = {}
            # fre_dict[attri2] = {}
        for i in range(len(correlated_full)):
            candidate_full[i] = copy.deepcopy(correlated_full[i])
            for j,v in candidate_full[i].items():
                if j in fre_dict.keys():
                    candidate_full[i][j] = list()
                    candidate_full[i][j].append(correlated_full[i][j])
                    fre_dict[j][correlated_full[i][j]] = 1
        for i in range(matrix_length):
            for j in range(matrix_length):
                if len(map_matrix[(i,j)]) != 0:
                    for k in range(len(map_matrix[(i,j)])):
                        first_tuple_index = map_matrix[(i,j)][k][0]
                        second_tuple_index = map_matrix[(i,j)][k][1]
                        if first_tuple_index == second_tuple_index:
                            continue
                        self.fix_tuple_pair(correlated_full,candidate_full, fre_dict,first_tuple_index, second_tuple_index)
        return candidate_full,fre_dict


    def fix_tuple_pair(self,correlated_full,candidate_full,fre_dict,first_idx,second_idx):
        fst_tup = correlated_full[first_idx]
        sec_tup = correlated_full[second_idx]
        err_sig = True
        for dc_it in range(len(self.dc.predicates)):
            attri1 = self.dc.predicates[dc_it].components[0]
            #若第二个属性为常数
            if self.dc.predicates[dc_it].property[1] =='constant':
                attri2 = 'const-attri'
                sec_tup[attri2] = float(self.dc.predicates[dc_it].components[1])
            else:
                attri2 = self.dc.predicates[dc_it].components[1]
            pred_oper = self.dc.predicates[dc_it].opt
            if self.judge_oper(fst_tup[attri1],sec_tup[attri2],pred_oper):
                continue
            else:
                err_sig = False#有一条predicate不满足，则说明这个tuple pair没有dc冲突
                break
        if err_sig == True:
            for i in range(len(self.dc.predicates)):
                attri1 = self.dc.predicates[i].components[0]
                if self.dc.predicates[i].property[1] == 'constant':
                    attri2 = 'const-attri'
                    sec_tup[attri2] = self.dc.predicates[i].components[1]
                else:
                    attri2 = self.dc.predicates[i].components[1]
                pred_oper = self.dc.predicates[i].opt
                candid_range1 = self.invert_atom(fst_tup[attri1],sec_tup[attri2],pred_oper)
                self.cal_frequence(fre_dict[attri1], candid_range1)
                candidate_full[first_idx][attri1].append(candid_range1)
                if self.dc.predicates[i].property[1] != 'constant':
                    candid_range2 = self.conform_atom(sec_tup[attri2],fst_tup[attri1],pred_oper)
                    self.cal_frequence(fre_dict[attri2], candid_range2)
                    candidate_full[second_idx][attri2].append(candid_range2)
        elif err_sig == False:
            for i in range(len(self.dc.predicates)):
                attri1 = self.dc.predicates[i].components[0]
                self.cal_frequence(fre_dict[attri1], fst_tup[attri1])
                candidate_full[first_idx][attri1].append(fst_tup[attri1])
                if self.dc.predicates[i].property[1] != 'constant':
                    attri2 = self.dc.predicates[i].components[1]
                    self.cal_frequence(fre_dict[attri2], sec_tup[attri2])
                    candidate_full[second_idx][attri2].append(sec_tup[attri2])
        pass


    def judge_oper(self,a,b,oper):
        if oper == ">=":
            if a >= b:return True
            else:return False
        elif oper == ">":
            if a > b:return True
            else:return False
        elif oper == "=":
            if a == b:return True
            else:return False
        elif oper == "<>":
            if a != b:return True
            else:return False
        elif oper == "<":
            if a < b:return True
            else:return False
        elif oper == "<=":
            if a <= b:return True
            else:return False


    def invert_atom(self,a,b,oper):
        if oper == '>=':
            return "<"+str(b)
        elif oper == '>':
            return "<="+str(b)
        elif oper == '=':
            return "!="+str(b)
        elif oper == '<>':
            return "=="+str(b)
        elif oper == '<':
            return ">="+str(b)
        elif oper == '<=':
            return ">"+str(b)


    def conform_atom(self,a,b,oper):
        if oper == '>=':
            return ">="+str(b)
        elif oper == '>':
            return ">"+str(b)
        elif oper == '=':
            return "=="+str(b)
        elif oper == '<>':
            return "!="+str(b)
        elif oper == '<':
            return "<"+str(b)
        elif oper == '<=':
            return "<="+str(b)


    def cal_prob(self,candidate_full,freq_dict):
        num = len(candidate_full)
        dc_attri = self.dc.schema
        for i in range(num):
            for attri in dc_attri:
                dict = {}
                for j in range(len(candidate_full[i][attri])):
                    dict[candidate_full[i][attri][j]] = candidate_full[i][attri].count(candidate_full[i][attri][j]) / len(candidate_full[i][attri])
                candidate_full[i][attri] = dict


    def cal_frequence(self,fre_dict,key):
        if key not in fre_dict.keys():
            fre_dict[key] = 1
        else:
            fre_dict[key] = fre_dict[key] + 1


    def detectAndrepair_repetition(self,map_matrix,correlated_full):
        attribute = self.dc.schema[0]
        repetition_fre_dict = {}
        repetition_ind_dict = {}
        matrix_length = int(math.sqrt(len(map_matrix)))
        for i in range(matrix_length):
            for j in range(matrix_length):
                if len(map_matrix[(i,j)]) == 0:
                    continue
                for list_item in map_matrix[(i,j)]:
                    value_check = correlated_full[list_item[0]][attribute]
                    if value_check in repetition_fre_dict.keys():
                        repetition_fre_dict[value_check] = repetition_fre_dict[value_check]+1
                    else:
                        repetition_fre_dict[value_check] = 1
                    if value_check in repetition_ind_dict.keys():
                        repetition_ind_dict[value_check].add(list_item[0])
                        repetition_ind_dict[value_check].add(list_item[1])
                    else:
                        repetition_ind_dict[value_check] = set()
                        repetition_ind_dict[value_check].add(list_item[0])
                        repetition_ind_dict[value_check].add(list_item[1])
        #求出每个元素的重复次数
        for key,value in repetition_fre_dict.items():
            value = int(math.sqrt(value))
            value = value+1
            repetition_fre_dict[key] = value
        candidate_full = {} #存放修改后的结果
        for i in range(len(correlated_full)):
            candidate_full[i] = copy.deepcopy(correlated_full[i])
        # 如果去重的属性为字符串
        if isinstance(correlated_full[0][attribute], str):
            for key, value_set in repetition_ind_dict.items():
                i = 0
                for ind in value_set:
                    candidate_full[ind][attribute] = candidate_full[ind][attribute]+"_wx_"+str(i)
                    i = i+1
        # 如果去重的属性为数字,简单依次变为从-1开始的负数
        else:
            i = -1
            for key, value_set in repetition_ind_dict.items():
                j = 0
                for ind in value_set:
                    if j == 0:continue
                    else:
                        candidate_full[ind][attribute] = i
                        i = i - 1
                        j = j + 1
        return candidate_full



class CleanJoin:
    def __init__(self,dirty_answer,dataset1,dataset2,join):
        self.dirty_answer = dirty_answer
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.join = join
        self.fix_table1()

    def fix_table1(self):
        dirty_answer1 = copy.deepcopy(self.dirty_answer)
        table1_attri = list(copy.deepcopy(self.dataset1.dict[0].index))
        table1_attri.append('original-index')
        for i in range(len(self.dirty_answer)):
            for index,value in self.dirty_answer[i].items():
                if index not in table1_attri:
                    del(dirty_answer1[i][index])
        cleansigma1 = CleanSigma(dataset=self.dataset1,answer=dirty_answer1,fd=self.join.fd1)
        clean1 = {}
        table1_attri.remove('original-index')
        table1_project_attri = []
        for i in range(len(self.dirty_answer[0].index)):
            if self.dirty_answer[0].index[i] in table1_attri:
                table1_project_attri.append(self.dirty_answer[0].index[i])
        index = 0
        relax_merge = {}
        for i in range(len(cleansigma1.rela_ans_tup)):
            for j in cleansigma1.rela_ans_tup[i].keys():
                relax_merge[index] = cleansigma1.rela_ans_tup[i][j]
                index = index + 1
        for i in range(len(cleansigma1.rela_extr_tup)):
            for j in cleansigma1.rela_ans_tup[i].keys():
                relax_merge[index] = cleansigma1.rela_extr_tup[i][j]
                index = index + 1
        for key in relax_merge.keys():
            for ind in relax_merge[key].index:
                if ind not in table1_project_attri:
                    del(relax_merge[key][ind])
        self.deduplicate(relax_merge)
        i = 0
        relax_merge1 = {}
        for key in relax_merge.keys():
            relax_merge1[i] = relax_merge[key]
            i = i+1
        dataset_temp = copy.deepcopy(self.dataset1)
        dataset_temp.dict = relax_merge1
        select_temp = daisy.Select(select_attribute=table1_project_attri,from_table=self.dataset1.name,where_caluse='')
        join_temp = daisy.Join(table1=self.join.table1,table2=self.join.table2,join_key=self.join.join_key,
                               project_attri=self.join.project_attri,select=select_temp,
                               fd1=self.join.fd1,fd2=self.join.fd2)
        if (join_temp.join_key != join_temp.fd1.lhs) and (join_temp.join_key != join_temp.fd1.rhs):
            join_after_relax = daisy.do_join(dataset_temp,self.dataset2,join_temp)
        else:
            join_after_relax = self.probability_join(dataset_temp,self.dataset2,join_temp)
        pass


    def deduplicate(self,dataset):
        for key in dataset.keys():
            for index in dataset[key].index:
                if type(dataset[key][index]).__name__ == 'dict':
                    for key1 in dataset[key][index].keys():
                        if dataset[key][index][key1] == 1.0:
                            dataset[key][index] = key1
        merge_dataset = {}
        record_duplicate = set()
        for key1 in dataset.keys():
            for key2 in dataset.keys():
                issame = True
                if key1 >= key2:
                    continue
                for ind in dataset[key2].index:
                    if dataset[key1][ind] != dataset[key2][ind]:
                        issame = False
                        break
                if issame:
                    record_duplicate.add(key2)
        for i in record_duplicate:
            del(dataset[i])
        pass


    def probability_join(self,dataset1,dataset2,join):
        select_result = dataset1.dict
        join_key = join.join_key
        project_attri = []
        project_attri = copy.deepcopy(join.project_attri)
        project_attri.remove(join_key)
        str1 = str(join_key+'1')
        str2 = str(join_key+'2')
        project_attri.append(str1)
        project_attri.append(str2)
        result = {}
        index = 0
        list = []
        for i in range(len(project_attri)):
            list.append(0)
        for i in range(len(select_result)):
            for j in range(len(dataset2.dict)):
                if type(select_result[i][join_key]).__name__ != 'dict':
                    if select_result[i][join_key] == dataset2.dict[j][join_key]:
                        result[index] = pd.Series(list, index=project_attri)
                        for k in range(len(project_attri)):
                            if project_attri[k] in select_result[0].index:
                                result[index][project_attri[k]] = select_result[i][project_attri[k]]
                            elif project_attri[k] in dataset2.dict[0].index:
                                result[index][project_attri[k]] = dataset2.dict[j][project_attri[k]]
                            elif k==len(project_attri)-2:
                                result[index][project_attri[k]] = select_result[i][join_key]
                            elif k==len(project_attri)-1:
                                result[index][project_attri[k]] = dataset2.dict[j][join_key]
                        index = index + 1
                else:
                    if dataset2.dict[j][join_key] in select_result[i][join_key].keys():
                        result[index] = pd.Series(list, index=project_attri)
                        for k in range(len(project_attri)):
                            if project_attri[k] in select_result[0].index:
                                result[index][project_attri[k]] = select_result[i][project_attri[k]]
                            elif project_attri[k] in dataset2.dict[0].index:
                                result[index][project_attri[k]] = dataset2.dict[j][project_attri[k]]
                            elif k == len(project_attri) - 2:
                                result[index][project_attri[k]] = select_result[i][join_key]
                            elif k == len(project_attri) - 1:
                                result[index][project_attri[k]] = dataset2.dict[j][join_key]
                        index = index + 1
        self.trigger_violation2(dataset2,result,join.fd2)
        return result


    def trigger_violation2(self,dataset,result,fd):
        temp_result = copy.deepcopy(result)
        lhs = fd.lhs
        rhs = fd.rhs
        attri_ = ''
        for ind in temp_result[0].index:
            if ind == lhs or ind == rhs:
                attri_ = ind
                attri = attri_
        if attri_ == '':
            attri_ = temp_result[0].index[-1]
            attri = attri_[:len(attri_)-1]
        if attri == lhs:
            attri1 = rhs
        else:
            attri1 = lhs
        for key in temp_result.keys():
            temp_result[key][attri_] = self.fix_probability(temp_result[key],result,temp_result[key][attri_],
                                                            attri_,attri,attri1,dataset.dict)
        for i in range(len(temp_result)):
            result[i] = temp_result[i]
        pass

    def fix_probability(self,temp_series,result,value,attri_,attri,attri1,dict):
        index = len(result)
        attri_project = list(result[0].index)
        dict_temp = {}
        remain_list = []
        sum = 0
        for key in dict.keys():
            if dict[key][attri] == value:
                value1 = dict[key][attri1]
                loc = key
        for key in dict.keys():
            if dict[key][attri1] == value1:
                if key != loc:
                    remain_list.append(key)
                if dict[key][attri] not in dict_temp.keys():
                    dict_temp[dict[key][attri]] = {}
                    dict_temp[dict[key][attri]] = 1
                    sum = sum + 1
                else:
                    dict_temp[dict[key][attri]] = dict_temp[dict[key][attri]] + 1
                    sum = sum + 1
        if len(dict_temp) == 1:
            return value
        else:
            for key in dict_temp.keys():
                dict_temp[key] = dict_temp[key] / sum
            for i in remain_list:
                result[index] = copy.deepcopy(temp_series)
                for j in attri_project:
                    if j == attri_:
                        result[index][j] = dict_temp
                    elif j in dict[0].index:
                        result[index][j] = dict[i][j]

            return dict_temp


class MultiCleanSigma:

    def __init__(self,dataset,cleansigma_list,fd_list=None,dc_list=None):
        self.dataset = dataset
        self.cleansigma_list = cleansigma_list
        if fd_list != None:
            self.fd_list = fd_list
            self.multiRelaxFD(dataset.dict)
        elif dc_list != None:
            self.dc_list = dc_list
            self.multiRelaxDC(dataset.dict)


    def multiRelaxFD(self,dict):
        result_list = []
        for item in self.cleansigma_list:
            temp1 = item.rela_ans_tup
            temp2 = item.rela_extr_tup
            temp = self.merge_dict(temp1,temp2)
            result_list.append(temp)
        fd_attri_num = {}
        overlap_attri = []
        for item in self.fd_list:
            if item.lhs in fd_attri_num.keys():
                fd_attri_num[item.lhs] = fd_attri_num[item.lhs] + 1
            else:
                fd_attri_num[item.lhs] = 1
            if item.rhs in fd_attri_num.keys():
                fd_attri_num[item.rhs] = fd_attri_num[item.rhs] + 1
            else:
                fd_attri_num[item.rhs] = 1
        for key in fd_attri_num.keys():
            if fd_attri_num[key] > 1:
                overlap_attri.append(key)
        if len(overlap_attri)==0:
            self.merge_result_list(result_list)
        else:
            for attri in overlap_attri:
                relevant_fd = self.get_revelant_fd(self.fd_list,attri) #revelant_fd 是 list
                revelant_attri = self.get_fd_revelant_attri(relevant_fd,attri) #revelant_attri 是 set
                self.fix_res_fd(dict,result_list,attri,revelant_attri)
                self.merge_result_list(result_list)
                pass

    def multiRelaxDC(self,dict):
        result_list = self.trans_res_candidate_form(self.cleansigma_list)
        overlap_attri = []
        # a = set(self.dc_list[0].schema)
        # for i in range(1,len(self.dc_list)):
        #     b = set(self.dc_list[i].schema)
        #     a = a & b
        a = set()
        for i in range(len(self.dc_list)):
            for j in range(i+1,len(self.dc_list)):
                for item in self.dc_list[i].schema:
                    if item in a:
                        continue
                    if item in self.dc_list[j].schema:
                        a.add(item)
        overlap_attri = list(a)
        #待验证
        if len(overlap_attri)==0:
            self.merge_result_list(result_list)
        else:
            for attri in overlap_attri:
                relevant_attri = self.get_dc_revelant_attri(attri) #revelant_attri 是 set
                self.fix_res_dc(dict,result_list,attri,relevant_attri)
                self.merge_result_list(result_list)
            multi_dc_result = self.final_result(result_list)
            self.relax_result = multi_dc_result


    def trans_res_candidate_form(self,cleansigma_list):
        result_list = []
        for i in range(len(cleansigma_list)):
            result_list.append(cleansigma_list[i].relax_full)
        for i in range(len(result_list)):
            attri_list = self.dc_list[i].schema
            for j in result_list[i]:
                original_index = result_list[i][j].name
                tmp_dic = {}
                for k in range(len(attri_list)):
                    tmp_series = copy.deepcopy(self.dataset.dict[original_index])
                    tmp_series[attri_list[k]] = result_list[i][j][attri_list[k]]
                    string = 'candidate' + str(k+1)
                    tmp_dic[string] = tmp_series
                result_list[i][j] = tmp_dic
        return result_list


    def get_dc_revelant_attri(self,attri):
        attri_set = set()
        for i in range(len(self.dc_list)):
            if attri in self.dc_list[i].schema:
                for j in range(len(self.dc_list[i].schema)):
                    if self.dc_list[i].schema[j] == attri:
                        continue
                    else:
                        attri_set.add(self.dc_list[i].schema[j])
        return attri_set


    def merge_dict(self,dict1,dict2):
        dict = {}
        index = 0
        for i in range(len(dict1)):
            dict[index] = dict1[i]
            index = index + 1
        for i in range(len(dict2)):
            dict[index] = dict2[i]
            index = index + 1
        return dict


    def get_revelant_fd(self,fd_list,attri):
        list = []
        for item in fd_list:
            if item.lhs == attri or item.rhs == attri:
                list.append(item)
        return list


    def get_fd_revelant_attri(self,fd_list,attri):
        revelant_attri = set()
        for item in fd_list:
            if item.lhs != attri:
                revelant_attri.add(item.lhs)
            if item.rhs != attri:
                revelant_attri.add(item.rhs)
        return revelant_attri


    def fix_res_fd(self,dict,result_list,attri,revelant_attri):
        # correlate_tup_index = []
        for item in result_list:
            correlate_tup_index = []
            for key1 in item.keys():
                correlate_tup_index.append(item[key1]['candidate1'].name)
            for key1 in item.keys():
                for key2 in item[key1].keys():
                    if type(item[key1][key2][attri]).__name__ == 'dict':
                        for key3 in item[key1][key2][attri].keys():
                            value = key3
                            item[key1][key2][attri][key3] = self.cal_multi_probability_fd(item[key1][key2].name,dict,
                                                                                 correlate_tup_index,attri,
                                                                                 revelant_attri,value)
        pass


    def cal_multi_probability_fd(self,index,dict,correlate_index,attri,revelant_attri,value):
        sum = 0
        num = 0
        for ind in correlate_index:
            flag = True
            for ele in revelant_attri:
                if dict[ind][ele] == dict[index][ele]:
                    if flag == True:
                        sum = sum + 1
                        flag = False
                    if dict[ind][attri] == value:
                        num = num + 1
                        break
        return num/sum


    def fix_res_dc(self,dict,result_list,attri,revelant_attri):
        self.merge_result_list(result_list)
        #拷贝一个原结果作为参照
        result_list0_copy = copy.deepcopy(result_list[0])
        for key1 in result_list0_copy.keys():
            for key2 in result_list0_copy[key1].keys():
                for key3 in result_list0_copy[key1][key2].keys():
                    result_list0_copy[key1][key2][key3] = copy.deepcopy(result_list[0][key1][key2][key3])
        #对概率进行处理
        for item in result_list[0].keys():
            for key1 in result_list[0][item].keys():
                    if type(result_list[0][item][key1][attri]).__name__ == 'dict':
                        for key2 in result_list[0][item][key1][attri].keys():
                            value = key2
                            result_list[0][item][key1][attri][key2] = self.cal_multi_probability_dc(result_list[0][item][key1], result_list0_copy,
                                                                                        attri,revelant_attri,value)
        pass


    def cal_multi_probability_dc(self,candidate,result_list_copy,attri,revelant_attri,value):
        frequent_dict = {}
        for key0 in result_list_copy.keys():
            item = result_list_copy[key0]
            for key1 in item.keys():
                if type(item[key1][attri]).__name__ == 'dict':#对于修改的是关键属性的candidate
                    for i in revelant_attri: #对比相关属性，看是否有相同项
                        if item[key1][i] == candidate[i]: #有相关属性相同
                            for key2 in item[key1][attri].keys():
                                if key2 in frequent_dict.keys():
                                    frequent_dict[key2] = frequent_dict[key2] + item[key1][attri][key2]
                                else:
                                    frequent_dict[key2] = item[key1][attri][key2]
                            break
        sum = 0
        for key in frequent_dict.keys():
            sum = sum + frequent_dict[key]
        try:
            ratio = frequent_dict[value] / sum
        except:
            ratio = 0
        return ratio


    def merge_result_list(self,result_list):
        for i in range(len(result_list)):
            if len(result_list[i])==0:
                continue
            j = i + 1
            while j < len(result_list):
                for key1 in result_list[i].keys():
                    index = result_list[i][key1]['candidate1'].name
                    for key11 in result_list[j].keys():
                        if result_list[j][key11]['candidate1'].name == index:
                            self.merge_candidate(result_list[i][key1],result_list[j][key11])
                            del result_list[j][key11]
                            break
                j = j+1
        index = len(result_list[0])
        i = 1
        while i < len(result_list):
            if len(result_list[i]) > 0:
                for key in result_list[i].keys():
                    result_list[0][index] = result_list[i][key]
                    index = index + 1
            i = i + 1
        pass


    def merge_candidate(self,dict1,dict2):
        dict1_length = len(dict1)
        add_dict = {}
        for key2 in dict2.keys():
            add_flag = True
            for key1 in dict1.keys():
                if dict1[key1].equals(dict2[key2]):
                    add_flag = False
                    break
            if add_flag == True:
                add_dict[key2] = dict2[key2]
        index = dict1_length+1
        for key in add_dict.keys():
            strr = 'candidate'+str(index)
            dict1[strr] = add_dict[key]
            index = index + 1


    def final_result(self,result_list):
        result = result_list[0]
        for key in result.keys():
            self.adjust(result[key])#将每行的各个candidate调整为每个candidate改变的属性都不重复的情况
        for key in result.keys():
            temp = self.integrate_candidate(result[key])
            result[key] = temp
        return result


    def adjust(self,row):
        del_list = []
        temp = {}
        for i in range(1,len(row)+1):
            if i in del_list:
                continue
            for attri in row['candidate'+str(i)].keys():
                if isinstance(row['candidate'+str(i)][attri],dict):
                    break
            for j in range(i+1,len(row)+1):
                if isinstance(row['candidate'+str(j)][attri],dict):
                    dict1 = row['candidate'+str(i)][attri]
                    dict2 = row['candidate'+str(j)][attri]
                    for key,value in dict2.items():
                        if key in dict1.keys():
                            dict1[key] = dict1[key]+value
                        else:
                            dict1[key] = value
                    del_list.append(j)
        for i in range(len(row)+1,0,-1):
            if i in del_list:
                del row['candidate'+str(i)]
        ind = 1
        for key in row.keys():
            temp['candidate'+str(ind)] = row[key]
            ind = ind + 1
        row = temp



    def integrate_candidate(self,row):
        row_final = copy.deepcopy(row['candidate1'])
        for key in row.keys():
            for attri in row[key].keys():
                if isinstance(row[key][attri],dict):
                    temp_value = self.get_max(row[key][attri])
                    row_final[attri] = temp_value
        return row_final


    def get_max(self,dict):
        new_dict = {}
        if isinstance(list(dict.keys())[0],str): #如果该属性是str类型
            for key in dict.keys():
                strr = key
                if strr[0] == '!' or strr[0] == '>' or strr[0] == '<':
                    new_dict[strr] = dict[key]
                else:
                    if strr[0] == '=': # ==key类型
                        pure_key = strr[2:len(strr)]
                        if pure_key in new_dict.keys():
                            new_dict[pure_key] = new_dict[pure_key] + dict[strr]
                        else:
                            new_dict[pure_key] = dict[strr]
                    else: # key类型
                        if strr in new_dict.keys():
                            new_dict[strr] = new_dict[strr] + dict[strr]
                        else:
                            new_dict[strr] = dict[strr]
        else:#如果该属性是num类型
            for key in dict.keys():
                strr = key
                if isinstance(strr,str):
                    if strr[0] == '!' or strr[0] == '>' or strr[0] == '<':
                        new_dict[strr] = dict[key]
                    else:
                        if strr[0] == '=':  # ==key类型
                            pure_key = strr[2:len(strr)]
                            try:
                                if '.' in pure_key: #若为float
                                    num_value = float(pure_key)
                                    pure_key = num_value
                                else: #若为int
                                    num_value = int(pure_key)
                                    pure_key = num_value
                            except:
                                num_value = -1
                                pure_key = num_value
                            if pure_key in new_dict.keys():
                                new_dict[pure_key] = new_dict[pure_key] + dict[strr]
                            else:
                                new_dict[pure_key] = dict[strr]
                else:
                    if strr in new_dict.keys():
                        new_dict[strr] = new_dict[strr] + dict[strr]
                    else:
                        new_dict[strr] = dict[strr]
        max_value = 0
        for key in new_dict.keys():
            if new_dict[key] > max_value:
                max_attri = key
                max_value = new_dict[key]
        return max_attri

