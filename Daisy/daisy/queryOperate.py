import copy
import pandas as pd

def do_select(dataset,select):
    if(len(select.where_clause) == 0):
        result = do_select_nowhere(dataset,select)
    elif(select.where_clause[0] in dataset.df.columns.tolist() and select.where_clause[2] in dataset.df.columns.tolist()):
        if (select.where_clause[1] == '>='):
            result = do_select_BGE_AttriAttri(dataset, select)
        elif (select.where_clause[1] == '>'):
            result = do_select_BT_AttriAttri(dataset, select)
        elif (select.where_clause[1] == '=='):
            result = do_select_EQU_AttriAttri(dataset, select)
        elif (select.where_clause[1] == '<>'):
            result = do_select_NEQ_AttriAttri(dataset, select)
        elif (select.where_clause[1] == '<'):
            result = do_select_LT_AttriAttri(dataset, select)
        elif (select.where_clause[1] == '<='):
            result = do_select_LSE_AttriAttri(dataset, select)
    elif(select.where_clause[0] in dataset.df.columns.tolist()):
        if(select.where_clause[1] == '>='):
            result = do_select_BGE_AttriCons(dataset,select)
        elif(select.where_clause[1] == '>'):
            result = do_select_BT_AttriCons(dataset, select)
        elif (select.where_clause[1] == '=='):
            result = do_select_EQU_AttriCons(dataset, select)
        elif (select.where_clause[1] == '<>'):
            result = do_select_NEQ_AttriCons(dataset, select)
        elif (select.where_clause[1] == '<'):
            result = do_select_LT_AttriCons(dataset, select)
        elif (select.where_clause[1] == '<='):
            result = do_select_LSE_AttriCons(dataset, select)
    elif(select.where_clause[2] in dataset.df.columns.tolist()):
        if (select.where_clause[1] == '>='):
            result = do_select_BGE_ConsAttri(dataset, select)
        elif (select.where_clause[1] == '>'):
            result = do_select_BT_ConsAttri(dataset, select)
        elif (select.where_clause[1] == '=='):
            result = do_select_EQU_ConsAttri(dataset, select)
        elif (select.where_clause[1] == '<>'):
            result = do_select_NEQ_ConsAttri(dataset, select)
        elif (select.where_clause[1] == '<'):
            result = do_select_LT_ConsAttri(dataset, select)
        elif (select.where_clause[1] == '<='):
            result = do_select_LSE_ConsAttri(dataset, select)
    return result

#------------------------无where语句--------------------------------------------

def do_select_nowhere(dataset,select):
    project = select.select_attribute
    if project[-1] == 'original-index':
        del project[-1]
    result_dict = {}
    i = 0
    for key in dataset.dict.keys():
        result_dict[i] = dataset.dict[key][project]

        i = i+1
    return result_dict

#-------------------------属性-属性 类型----------------------------------------

def do_select_BGE_AttriAttri(dataset, select):
    project = select.select_attribute
    project.append('original-index')
    attri1 = select.where_clause[0]
    attri2 = select.where_clause[2]
    result_dict = {}
    index = 0
    for i in range(dataset.row):
        if (dataset.dict[i][attri1] >= dataset.dict[i][attri2]):
            temp_series = copy.deepcopy(dataset.dict[i])
            temp_series['original-index'] = i
            result_dict[index] = temp_series
            index = index + 1
    for i in range(index):
        result_dict[i] = result_dict[i][project]
    del select.select_attribute[-1]
    return result_dict


def do_select_BT_AttriAttri(dataset, select):
    project = select.select_attribute
    project.append('original-index')
    attri1 = select.where_clause[0]
    attri2 = select.where_clause[2]
    result_dict = {}
    index = 0
    for i in range(dataset.row):
        if (dataset.dict[i][attri1] > dataset.dict[i][attri2]):
            temp_series = copy.deepcopy(dataset.dict[i])
            temp_series['original-index'] = i
            result_dict[index] = temp_series
            index = index + 1
    for i in range(index):
        result_dict[i] = result_dict[i][project]
    del select.select_attribute[-1]
    return result_dict


def do_select_EQU_AttriAttri(dataset, select):
    project = select.select_attribute
    project.append('original-index')
    attri1 = select.where_clause[0]
    attri2 = select.where_clause[2]
    result_dict = {}
    index = 0
    for i in range(dataset.row):
        if (dataset.dict[i][attri1] == dataset.dict[i][attri2]):
            temp_series = copy.deepcopy(dataset.dict[i])
            temp_series['original-index'] = i
            result_dict[index] = temp_series
            index = index + 1
    for i in range(index):
        result_dict[i] = result_dict[i][project]
    del select.select_attribute[-1]
    return result_dict


def do_select_NEQ_AttriAttri(dataset, select):
    project = select.select_attribute
    project.append('original-index')
    attri1 = select.where_clause[0]
    attri2 = select.where_clause[2]
    result_dict = {}
    index = 0
    for i in range(dataset.row):
        if (dataset.dict[i][attri1] != dataset.dict[i][attri2]):
            temp_series = copy.deepcopy(dataset.dict[i])
            temp_series['original-index'] = i
            result_dict[index] = temp_series
            index = index + 1
    for i in range(index):
        result_dict[i] = result_dict[i][project]
    del select.select_attribute[-1]
    return result_dict


def do_select_LT_AttriAttri(dataset, select):
    project = select.select_attribute
    project.append('original-index')
    attri1 = select.where_clause[0]
    attri2 = select.where_clause[2]
    result_dict = {}
    index = 0
    for i in range(dataset.row):
        if (dataset.dict[i][attri1] < dataset.dict[i][attri2]):
            temp_series = copy.deepcopy(dataset.dict[i])
            temp_series['original-index'] = i
            result_dict[index] = temp_series
            index = index + 1
    for i in range(index):
        result_dict[i] = result_dict[i][project]
    del select.select_attribute[-1]
    return result_dict


def do_select_LSE_AttriAttri(dataset, select):
    project = select.select_attribute
    project.append('original-index')
    attri1 = select.where_clause[0]
    attri2 = select.where_clause[2]
    result_dict = {}
    index = 0
    for i in range(dataset.row):
        if (dataset.dict[i][attri1] <= dataset.dict[i][attri2]):
            temp_series = copy.deepcopy(dataset.dict[i])
            temp_series['original-index'] = i
            result_dict[index] = temp_series
            index = index + 1
    for i in range(index):
        result_dict[i] = result_dict[i][project]
    del select.select_attribute[-1]
    return result_dict

#-------------------------属性-常数 类型----------------------------------------

def do_select_BGE_AttriCons(dataset, select):
    project = select.select_attribute
    project.append('original-index')
    attri = select.where_clause[0]
    cons = select.where_clause[2]
    result_dict = {}
    index = 0
    for i in range(dataset.row):
        if (dataset.dict[i][attri] >= cons):
            temp_series = copy.deepcopy(dataset.dict[i])
            temp_series['original-index'] = i
            result_dict[index] = temp_series
            index = index + 1
    for i in range(index):
        result_dict[i] = result_dict[i][project]
    del select.select_attribute[-1]
    return result_dict


def do_select_BT_AttriCons(dataset, select):
    project = select.select_attribute
    project.append('original-index')
    attri = select.where_clause[0]
    cons = select.where_clause[2]
    result_dict = {}
    index = 0
    for i in range(dataset.row):
        if (dataset.dict[i][attri] > cons):
            temp_series = copy.deepcopy(dataset.dict[i])
            temp_series['original-index'] = i
            result_dict[index] = temp_series
            index = index + 1
    for i in range(index):
        result_dict[i] = result_dict[i][project]
    del select.select_attribute[-1]
    return result_dict


def do_select_EQU_AttriCons(dataset, select):
    project = select.select_attribute
    project.append('original-index')
    attri = select.where_clause[0]
    cons = select.where_clause[2]
    result_dict = {}
    index = 0
    for i in range(dataset.row):
        if (dataset.dict[i][attri] == cons):
            temp_series = copy.deepcopy(dataset.dict[i])
            temp_series['original-index'] = i
            result_dict[index] = temp_series
            index = index + 1
    for i in range(index):
        result_dict[i] = result_dict[i][project]
    del select.select_attribute[-1]
    return result_dict


def do_select_NEQ_AttriCons(dataset, select):
    project = select.select_attribute
    project.append('original-index')
    attri = select.where_clause[0]
    cons = select.where_clause[2]
    result_dict = {}
    index = 0
    for i in range(dataset.row):
        if (dataset.dict[i][attri] != cons):
            temp_series = copy.deepcopy(dataset.dict[i])
            temp_series['original-index'] = i
            result_dict[index] = temp_series
            index = index + 1
    for i in range(index):
        result_dict[i] = result_dict[i][project]
    del select.select_attribute[-1]
    return result_dict


def do_select_LT_AttriCons(dataset, select):
    project = select.select_attribute
    project.append('original-index')
    attri = select.where_clause[0]
    cons = select.where_clause[2]
    result_dict = {}
    index = 0
    for i in range(dataset.row):
        if (dataset.dict[i][attri] < cons):
            temp_series = copy.deepcopy(dataset.dict[i])
            temp_series['original-index'] = i
            result_dict[index] = temp_series
            index = index + 1
    for i in range(index):
        result_dict[i] = result_dict[i][project]
    del select.select_attribute[-1]
    return result_dict

def do_select_LSE_AttriCons(dataset, select):
    project = select.select_attribute
    project.append('original-index')
    attri = select.where_clause[0]
    cons = select.where_clause[2]
    result_dict = {}
    index = 0
    for i in range(dataset.row):
        if (dataset.dict[i][attri] <= cons):
            temp_series = copy.deepcopy(dataset.dict[i])
            temp_series['original-index'] = i
            result_dict[index] = temp_series
            index = index + 1
    for i in range(index):
        result_dict[i] = result_dict[i][project]
    del select.select_attribute[-1]
    return result_dict

#-------------------------常数-属性 类型----------------------------------------

def do_select_BGE_ConsAttri(dataset, select):
    project = select.select_attribute
    project.append('original-index')
    attri = select.where_clause[2]
    cons = select.where_clause[0]
    result_dict = {}
    index = 0
    for i in range(dataset.row):
        if (cons >= dataset.dict[i][attri]):
            temp_series = copy.deepcopy(dataset.dict[i])
            temp_series['original-index'] = i
            result_dict[index] = temp_series
            index = index + 1
    for i in range(index):
        result_dict[i] = result_dict[i][project]
    del select.select_attribute[-1]
    return result_dict


def do_select_BT_ConsAttri(dataset, select):
    project = select.select_attribute
    project.append('original-index')
    attri = select.where_clause[2]
    cons = select.where_clause[0]
    result_dict = {}
    index = 0
    for i in range(dataset.row):
        if (cons > dataset.dict[i][attri]):
            temp_series = copy.deepcopy(dataset.dict[i])
            temp_series['original-index'] = i
            result_dict[index] = temp_series
            index = index + 1
    for i in range(index):
        result_dict[i] = result_dict[i][project]
    del select.select_attribute[-1]
    return result_dict


def do_select_LSE_ConsAttri(dataset, select):
    project = select.select_attribute
    project.append('original-index')
    attri = select.where_clause[2]
    cons = select.where_clause[0]
    result_dict = {}
    index = 0
    for i in range(dataset.row):
        if (cons <= dataset.dict[i][attri]):
            temp_series = copy.deepcopy(dataset.dict[i])
            temp_series['original-index'] = i
            result_dict[index] = temp_series
            index = index + 1
    for i in range(index):
        result_dict[i] = result_dict[i][project]
    del select.select_attribute[-1]
    return result_dict


def do_select_LT_ConsAttri(dataset, select):
    project = select.select_attribute
    project.append('original-index')
    attri = select.where_clause[2]
    cons = select.where_clause[0]
    result_dict = {}
    index = 0
    for i in range(dataset.row):
        if (cons < dataset.dict[i][attri]):
            temp_series = copy.deepcopy(dataset.dict[i])
            temp_series['original-index'] = i
            result_dict[index] = temp_series
            index = index + 1
    for i in range(index):
        result_dict[i] = result_dict[i][project]
    del select.select_attribute[-1]
    return result_dict


def do_select_NEQ_ConsAttri(dataset, select):
    project = select.select_attribute
    project.append('original-index')
    attri = select.where_clause[2]
    cons = select.where_clause[0]
    result_dict = {}
    index = 0
    for i in range(dataset.row):
        if (cons != dataset.dict[i][attri]):
            temp_series = copy.deepcopy(dataset.dict[i])
            temp_series['original-index'] = i
            result_dict[index] = temp_series
            index = index + 1
    for i in range(index):
        result_dict[i] = result_dict[i][project]
    del select.select_attribute[-1]
    return result_dict


def do_select_EQU_ConsAttri(dataset, select):
    project = select.select_attribute
    project.append('original-index')
    attri = select.where_clause[2]
    cons = select.where_clause[0]
    result_dict = {}
    index = 0
    for i in range(dataset.row):
        if (cons == dataset.dict[i][attri]):
            temp_series = copy.deepcopy(dataset.dict[i])
            temp_series['original-index'] = i
            result_dict[index] = temp_series
            index = index + 1
    for i in range(index):
        result_dict[i] = result_dict[i][project]
    del select.select_attribute[-1]
    return result_dict


#-------------------------Join操作----------------------
def do_join(dataset1,dataset2,join):
    select_result = do_select(dataset1,join.select)
    join_key = join.join_key
    project_attri = []
    project_attri = copy.deepcopy(join.project_attri)
    project_attri.append('original-index')
    result = {}
    index = 0
    list = []
    for i in range(len(project_attri)):
        list.append(0)
    for i in range(len(select_result)):
        for j in range(len(dataset2.dict)):
            if select_result[i][join_key] == dataset2.dict[j][join_key]:
                result[index] = pd.Series(list,index=project_attri)
                for k in range(len(join.project_attri)):
                    if join.project_attri[k] in select_result[0].index:
                        result[index][join.project_attri[k]] = select_result[i][join.project_attri[k]]
                    else:
                        result[index][join.project_attri[k]] = dataset2.dict[j][join.project_attri[k]]
                index = index + 1
    return result