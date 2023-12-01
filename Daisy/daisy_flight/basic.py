import numpy as np

def flight_basic(dict):
    #处理sched-dep-time和act-dep-time
    str1 = 'sched_dep_time'
    str2 = 'act_dep_time'
    for key in dict.keys():
        if isinstance(dict[key][str1],str):
            str_value = dict[key][str1]
            if ":" in str_value:
                index = str_value.find(':')
                if len(str_value) <= index+3:
                    str_temp = str_value[0:index + 3] + ' '+'a.m.'
                    dict[key][str1] = str_temp
                elif str_value[index+3] == ' ':
                    if str_value[index+4] == 'a':
                        str_temp = str_value[0:index+4]+'a.m.'
                        dict[key][str1] = str_temp
                    elif str_value[index+4] == 'p':
                        str_temp = str_value[0:index+4]+'p.m.'
                        dict[key][str1] = str_temp
                    else:
                        str_temp = str_value[0:index + 4] + 'a.m.'
                        dict[key][str1] = str_temp
                elif str_value[index+3] == 'a':
                    str_temp = str_value[0:index+3]+' '+'a.m.'
                    dict[key][str1] = str_temp
                elif str_value[index+3] == 'p':
                    str_temp = str_value[0:index+3]+' '+'p.m.'
                    dict[key][str1] = str_temp
                else:
                    str_temp = str_value[0:index + 3] + ' ' + 'a.m.'
                    dict[key][str1] = str_temp
        else:
            dict[key][str1] = 'empty'

        if isinstance(dict[key][str2],str):
            str_value = dict[key][str2]
            if ":" in str_value:
                index = str_value.find(':')
                if len(str_value) <= index+3:
                    str_temp = str_value[0:index + 3] + ' '+'a.m.'
                    dict[key][str1] = str_temp
                elif str_value[index+3] == ' ':
                    if str_value[index+4] == 'a':
                        str_temp = str_value[0:index+4]+'a.m.'
                        dict[key][str2] = str_temp
                    elif str_value[index+4] == 'p':
                        str_temp = str_value[0:index+4]+'p.m.'
                        dict[key][str2] = str_temp
                    else:
                        str_temp = str_value[0:index + 4] + 'a.m.'
                        dict[key][str2] = str_temp
                elif str_value[index+3] == 'a':
                    str_temp = str_value[0:index+3]+' '+'a.m.'
                    dict[key][str2] = str_temp
                elif str_value[index+3] == 'p':
                    str_temp = str_value[0:index+3]+' '+'p.m.'
                    dict[key][str2] = str_temp
                else:
                    str_temp = str_value[0:index + 3] + ' ' + 'a.m.'
                    dict[key][str2] = str_temp

        else:
            dict[key][str2] = 'empty'

        if ':' not in dict[key][str1]:
            if ':' in dict[key][str2]:
                dict[key][str1] = dict[key][str2]
        if ':' not in dict[key][str2]:
            if ':' in dict[key][str1]:
                dict[key][str2] = dict[key][str1]

    #处理sched-arr-time和act-arr-time
    str1 = 'sched_arr_time'
    str2 = 'act_arr_time'
    for key in dict.keys():
        if isinstance(dict[key][str1],str):
            str_value = dict[key][str1]
            if ":" in str_value:
                index = str_value.find(':')
                if len(str_value) <= index+3:
                    str_temp = str_value[0:index + 3] + ' '+'a.m.'
                    dict[key][str1] = str_temp
                elif str_value[index+3] == ' ':
                    if str_value[index+4] == 'a':
                        str_temp = str_value[0:index+4]+'a.m.'
                        dict[key][str1] = str_temp
                    elif str_value[index+4] == 'p':
                        str_temp = str_value[0:index+4]+'p.m.'
                        dict[key][str1] = str_temp
                    else:
                        str_temp = str_value[0:index + 4] + 'a.m.'
                        dict[key][str1] = str_temp
                elif str_value[index+3] == 'a':
                    str_temp = str_value[0:index+3]+' '+'a.m.'
                    dict[key][str1] = str_temp
                elif str_value[index+3] == 'p':
                    str_temp = str_value[0:index+3]+' '+'p.m.'
                    dict[key][str1] = str_temp
                else:
                    str_temp = str_value[0:index + 3] + ' ' + 'a.m.'
                    dict[key][str1] = str_temp
        else:
            dict[key][str1] = 'empty'

        if isinstance(dict[key][str2],str):
            str_value = dict[key][str2]
            if ":" in str_value:
                index = str_value.find(':')
                if len(str_value) <= index+3:
                    str_temp = str_value[0:index + 3] + ' '+'a.m.'
                    dict[key][str1] = str_temp
                elif str_value[index+3] == ' ':
                    if str_value[index+4] == 'a':
                        str_temp = str_value[0:index+4]+'a.m.'
                        dict[key][str2] = str_temp
                    elif str_value[index+4] == 'p':
                        str_temp = str_value[0:index+4]+'p.m.'
                        dict[key][str2] = str_temp
                    else:
                        str_temp = str_value[0:index + 4] + 'a.m.'
                        dict[key][str2] = str_temp
                elif str_value[index+3] == 'a':
                    str_temp = str_value[0:index+3]+' '+'a.m.'
                    dict[key][str2] = str_temp
                elif str_value[index+3] == 'p':
                    str_temp = str_value[0:index+3]+' '+'p.m.'
                    dict[key][str2] = str_temp
                else:
                    str_temp = str_value[0:index + 3] + ' ' + 'a.m.'
                    dict[key][str2] = str_temp

        else:
            dict[key][str2] = 'empty'

        if ':' not in dict[key][str1]:
            if ':' in dict[key][str2]:
                dict[key][str1] = dict[key][str2]
        if ':' not in dict[key][str2]:
            if ':' in dict[key][str1]:
                dict[key][str2] = dict[key][str1]
