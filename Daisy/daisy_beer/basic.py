import numpy as np

def beer_basic(dict):
    #处理ounce
    for key in dict.keys():
        str = dict[key]['ounces']
        str = str[0:4]
        dict[key]['ounces'] = float(str)
    #处理abv
    for key in dict.keys():
        str = dict[key]['abv']
        if '%' in str:
            str1 = str.replace('%','')
        else:
            str1 = str
        dict[key]['abv'] = float(str1)
    #处理ibu
    for key in dict.keys():
        if np.isnan(dict[key]['ibu']):
            dict[key]['ibu'] = -1
    #处理state空白
    for key in dict.keys():
        if isinstance(dict[key]['state'],float):
            dict[key]['state'] = 'wx'
    return dict
