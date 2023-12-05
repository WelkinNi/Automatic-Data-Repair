import pandas as pd
import numpy as np
import copy
# All operators used in DC
opts = ['!=', '<=', '>=', '=', '<', '>']
opt_sign = ['IQ', 'LTE', 'GTE', 'EQ', 'LT', 'GT']


# Decide whether there are opts in predicates
def contains_opt(pred):
    for i in range(len(opt_sign)):
        if pred.find(opt_sign[i]) != -1:
            return i
    return None


class DCRule():
    def __init__(self, dc_string, schema):
        self.variable = []
        self.predicates = []
        self.schema = schema
        self.resolve_dc(dc_string)

    def resolve_dc(self, dc_string):
        # A single denial constraint
        dc_preds = dc_string.split("&")
        # Fliter all variables
        for pred in dc_preds:
            opt_idx = contains_opt(pred)
            if opt_idx is None:
                self.variable.append(pred)
        # Analyzing the rest predicates
        for pred_idx in range(len(self.variable), len(dc_preds)):
            self.predicates.append(Predicate(dc_preds[pred_idx], self.schema, self.variable))
        return None
    def att_copy(self, DR):
        self.variable = copy.deepcopy(DR.variable)
        self.predicates = copy.deepcopy(DR.predicates)
        self.schema = copy.deepcopy(DR.schema)


class Predicate():
    def __init__(self, pred, schema, variables):
        self.vars = variables
        self.pred = pred
        self.schema = schema
        self.property = []
        self.components = []
        self.tuples = []
        self.opt = ""
        self.parser()

    def parser(self):
        opt_idx = contains_opt(self.pred)
        if opt_idx is not None:
            opt = opts[opt_idx]
            self.opt = opt
            opt_len = len(opt_sign[opt_idx])
            # jump to the string after opt
            compts = self.pred[opt_len:].split(",")
            for compt in compts:
                compt = compt.strip("( )\t\n\r")
                # Decide whether the term is attribute or constant value
                if compt.find(".") != -1 and compt.split(".")[0] in self.vars:
                    compt = compt.split(".")
                    self.tuples.append(compt[0])
                    self.property.append("attribute")
                    self.components.append(str(compt[1]))
                    if compt[1] not in self.schema:
                        raise Exception('Attribute not in schema')
                else:
                    self.property.append("constant")
                    self.components.append(str(compt.strip('"')))


        else:
            raise Exception('Cannot find operation in predicate')


if __name__ == "__main__":
    # Original: ∀t0∈clean.csv,t1∈clean.csv:¬[t0.ZipCode=t1.ZipCode∧t0.State≠t1.State]
    # Reading Format: t1&t2&EQ(t1.ZipCode,t2.ZipCode)&IQ(t1.State,t2.State)
    DR = DCRule(
        't1&t2&EQ(t1.HospitalName,t2.HospitalName)&EQ(t1.PhoneNumber,t2.PhoneNumber)&EQ(t1.HospitalOwner,t2.HospitalOwner)&IQ(t1.State,t2.State)',['State', 'PhoneNumber', 'HospitalName', 'HospitalOwner'])