import numpy as np

def hospital_basic(dict):
    #处理provider-number
    for key in dict.keys():
        if 'x' not in dict[key]['provider_number']:
            temp_value = int(dict[key]['provider_number'])
            dict[key]['provider_number'] = temp_value
        else:
            dict[key]['provider_number'] = temp_value

    #处理Zip
    for key in dict.keys():
        if 'x' not in dict[key]['zip']:
            temp_value = int(dict[key]['zip'])
            dict[key]['zip'] = temp_value
        else:
            dict[key]['zip'] = temp_value

    #处理Phone
    for key in dict.keys():
        if 'x' not in dict[key]['phone']:
            temp_value = float(dict[key]['phone'])
            dict[key]['phone'] = temp_value
        else:
            dict[key]['phone'] = temp_value

    #处理type
    for key in dict.keys():
        if dict[key]['type'] != 'acute care hospitals':
            dict[key]['type'] = 'acute care hospitals'

    #处理emergency-service
    for key in dict.keys():
        if 'x' in dict[key]['emergency_service']:
            if 'y' in dict[key]['emergency_service'] or 'e' in dict[key]['emergency_service'] or 's' in dict[key]['emergency_service']:
                dict[key]['emergency_service'] = 'yes'
            else:
                dict[key]['emergency_service'] = 'no'