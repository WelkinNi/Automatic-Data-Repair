# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import queue
import time
import argparse
import sys
import csv
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

class Vertex:
    def __init__(self, key, key1, type):
        self.id = key
        self.attr = key1
        self.type = type
        self.connectedTo = {}
        self.connectedQLT = {}

    def addNeighbor(self, nbr):
        if nbr in self.connectedTo:
            self.connectedTo[nbr] = self.connectedTo[nbr] + 1
        else:
            self.connectedTo[nbr] = 1
        self.connectedQLT[nbr] = 0

    def __str__(self):
        return str(self.id) + 'connectedTo' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return list(self.connectedTo.keys())

    def getId(self):
        return self.id

    def getAttr(self):
        return self.attr

    def getType(self):
        return self.type

    def getweight(self,nbr):
        return  self.connectedTo[nbr]


class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key, key1, type):
        if key not in self.vertList:
            self.numVertices = self.numVertices + 1
            newVertex = Vertex(key, key1, type)
            self.vertList[key] = newVertex
            return newVertex

    def getVertex(self,n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self, n):
        return  n in self.vertList

    def addEdge(self,f,t,const = 0):
        self.vertList[f].addNeighbor(self.vertList[t])

    def getVertices(self):
        return  self.vertList.keys()

    def __iter__(self):
        return  iter(self.vertList.values())



def BuildFDPatternGraph(D_path, constrains_path):
    g = Graph()
    my_dict = {}
    data = pd.read_csv(D_path)
    data = data.fillna("nan")
    data = data.astype(str)
    tot = len(data)
    f = open(constrains_path, encoding='utf-8')
    for line in f:
        lr = line.split("⇒")
        if lr[0].strip() in my_dict and my_dict[lr[0].strip()] == 1:
            my_dict[lr[1].strip()] = 1
            continue
        my_dict[lr[0].strip()] = 0
        my_dict[lr[1].strip()] = 1
    f.close()
    f = open(constrains_path, encoding='utf-8')
    for line in f:
        lr = line.split("⇒")
        left_data = data[lr[0].strip()].tolist()
        right_data = data[lr[1].strip()].tolist()
        uni_left_data = list(set(left_data))
        uni_right_data = list(set(right_data))
        # for i in range(0, len(uni_left_data)):
        #     if type(uni_left_data[i]) == float:
        #         uni_left_data[i] = ''
        # for i in range(0, len(uni_right_data)):
        #     if type(uni_right_data[i]) == float:
        #         uni_right_data[i] = ''
        # for i in range(0, len(left_data)):
        #     if type(left_data[i]) == float:
        #         left_data[i] = ''
        # for i in range(0, len(right_data)):
        #     if type(right_data[i]) == float:
        #         right_data[i] = ''
        for i in range(0, len(uni_left_data)):
            g.addVertex(str(uni_left_data[i]), lr[0].strip(), my_dict[lr[0].strip()])
        for i in range(0, len(uni_right_data)):
            g.addVertex(str(uni_right_data[i]), lr[1].strip(), my_dict[lr[1].strip()])
        for i in range(0, len(left_data)):
            g.addEdge(str(left_data[i]), str(right_data[i]))
        cnt = 0
    for v in g:
        for w in v.getConnections():
            v.connectedTo[w] = v.connectedTo[w]/tot
    for v in g:
        for w in v.getConnections():
            #print("( %s , %s )" % (v.getId(), w.getId()))
            cnt = cnt + 1
    f.close()
    return g


def dfs(g, root, vis):
    if len(root.getConnections()) == 0:
        return 0, 0
    if vis[root.attr] == 1:
        return 0, 0
    sup = 0
    num = 0
    vis[root.attr] = 1
    for v in root.getConnections():
        if vis[g.vertList[v.id].attr] == 0:
            sup += root.connectedTo[v]
            num += 1
            tmpa, tmpb = dfs(g, g.vertList[v.id], vis)
            sup += tmpa
            num += tmpb
            root.connectedQLT[v] = (tmpa + root.connectedTo[v])/(tmpb + 1)
    vis[root.attr] = 0
    return sup, num


def dfs1(g, root, vis):
    print(root.id)
    if len(root.getConnections()) == 0:
        return
    if vis[root.attr] == 1:
        return
    vis[root.attr] = 1
    for v in root.getConnections():
        print(root.connectedQLT[v])
        dfs1(g, g.vertList[v.id],vis)
    vis[root.attr] = 0


def ComputePatternQulity(g):
    vis = {}
    for v in g:
        if v.getType() == 0:
            for vv in g:
                vis[vv.attr] = 0
            dfs(g, v, vis)


def tr(G):
    GT = dict()
    for u in G.keys():
        GT[u] = GT.get(u, set())
    for u in G.keys():
        for v in G[u]:
            GT[v].add(u)
    return GT


def topoSort(G):
    res = []
    S = set()
    def dfs(G, u):
        if u in S:
            return
        S.add(u)
        for v in G[u]:
            if v in S:
                continue
            dfs(G, v)
        res.append(u)
    for u in G.keys():
        dfs(G, u)
    res.reverse()
    return res


def walk(G, s, S=None):
    if S is None:
        s = set()
    Q = []
    P = dict()
    Q.append(s)
    P[s] = None
    while Q:
        u = Q.pop()
        for v in G[u]:
            if v in P.keys() or v in S:
                continue
            Q.append(v)
            P[v] = P.get(v, u)
    return P

def BuildSCCGraghAndSort(constrains_path):
    sccg = Graph()
    G = {}
    f = open(constrains_path, encoding='utf-8')
    for line in f:
        lr = line.split("⇒")
        sccg.addVertex(lr[0].strip(), "", 0)
        sccg.addVertex(lr[1].strip(), "", 0)
        sccg.addEdge(lr[0].strip(), lr[1].strip())
    f.close()
    for v in sccg.vertList:
        tmp = set()
        for vv in sccg.vertList[v].getConnections():
            tmp.add(vv.id)
        G.update({v: tmp})
    print(G)
    seen = set()
    scc = []
    GT = tr(G)
    for u in topoSort(G):
        if u in seen:
            continue
        C = walk(GT, u, seen)
        seen.update(C)
        scc.append(sorted(list(C.keys())))
    tar = {}
    cnt = 0
    for li in scc:
        for ui in li:
            tar.update({ui: cnt})
        cnt += 1
    ret = {}
    for i in range(0, cnt):
        ret.update({i: set()})
    for li in G:
        for ui in G[li]:
            left = tar[li]
            right = tar[ui]
            if left != right:
                ret[left].add(right)
    indegree = []
    for i in range(0, cnt):
        indegree.append(0)
    for i in range(0, cnt):
        for j in ret[i]:
            indegree[j] += 1
    q = queue.Queue()
    for i in range(0, cnt):
        if indegree[i] == 0:
            q.put(i)
    count = 0
    order = []
    while count < cnt:
        count += 1
        ele = q.get()
        order.append(ele)
        for i in ret[ele]:
            indegree[i] -= 1
            if indegree[i] == 0:
                q.put(i)
    print(scc)
    print(ret)
    print(order)
    print(tar)
    return order, tar, scc, G


class tmporder:
    def __init__(self):
        left = ""
        right = ""
        lnum = 0
        rnum = 0


def OrderFDs(constrains_path, order, tar, scc, G):
    OrderedFDs = []
    f = open(constrains_path, encoding='utf-8')
    for line in f:
        lr = line.split("⇒")
        tmp = tmporder()
        tmp.lnum = tar[lr[0].strip()]
        tmp.rnum = tar[lr[1].strip()]
        tmp.left = lr[0].strip()
        tmp.right = lr[1].strip()
        OrderedFDs.append(tmp)
    OrderedFDs.sort(key=lambda x: x.rnum)
    OrderedFDs.sort(key=lambda x: x.lnum)
    for i in OrderedFDs:
        print(i.lnum,i.rnum)
    return OrderedFDs

def calDetPrecRec(pattern_expressions, dirty_path, clean_path):
    attrList = []
    dirty_dict = []
    with open(dirty_path, 'r') as f:
        reader = csv.DictReader(f, restval='nan')
        for line in reader:
            dirty_dict.append(line)
            attrList = list(line.keys())
    df = pd.read_csv(clean_path, header=0)
    df.columns = attrList
    # df.to_csv(clean_path, index=False)
    tot = 0
    correct_rec = 0
    with open(dirty_path, 'r', encoding='utf-8') as f:
        # global dirty cells: dirty_c 
        reader = csv.DictReader(f, restval='nan')
        cnt = 0
        for line in reader:
            for v in pattern_expressions[cnt]:
                if pattern_expressions[cnt][v] != dirty_dict[cnt][v]:
                    if (cnt, list(clean_df.columns).index(v)) in dirty_c:
                        correct_rec += 1
                tot += 1
            cnt += 1
    return correct_rec/tot, correct_rec/len(dirty_c)

def calRepPrec(pattern_expressions, dirty_path, clean_path):
    attrList = []
    dirty_dict = []
    with open(dirty_path, 'r') as f:
        reader = csv.DictReader(f, restval='nan')
        for line in reader:
            dirty_dict.append(line)
            attrList = list(line.keys())
            
    df = pd.read_csv(clean_path, header=0)
    df.columns = attrList
    # df.to_csv(clean_path, index=False)
    tot = 0
    correct = 0
    with open(clean_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, restval='nan')
        cnt = 0
        for line in reader:
            for v in pattern_expressions[cnt]:
                if PERFECTED:
                    if (cnt, list(clean_df.columns).index(v)) in dirty_c:
                        if pattern_expressions[cnt][v] != dirty_dict[cnt][v]:
                            if pattern_expressions[cnt][v] == line[v]:
                                correct += 1
                            tot += 1
                else:
                    if pattern_expressions[cnt][v] != dirty_dict[cnt][v]:
                        if pattern_expressions[cnt][v] == line[v]:
                            correct += 1
                        tot += 1
            cnt += 1
    return correct/tot

def calRepRec(pattern_expressions, dirty_path, clean_path):
    attrList = []
    with open(dirty_path, 'r') as f:
        reader = csv.DictReader(f, restval='nan')
        for line in reader:
            attrList = list(line.keys())
            break
    df = pd.read_csv(clean_path, header=0)
    df.columns = attrList
    # df.to_csv(clean_path, index=False)
    dirty_dict = []
    tot = 0
    correct = 0
    with open(dirty_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, restval='nan')
        for line in reader:
            dirty_dict.append(line)
    with open(clean_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, restval='nan')
        cnt = 0
        for line in reader:
            for v in dirty_dict[cnt]:
                if dirty_dict[cnt][v] != line[v]:
                    print(str((cnt, v)) + "| dirty: " + dirty_dict[cnt][v] + "| clean:" + line[v] + "| repaired:" + line[v])
                    try:
                        if pattern_expressions[cnt][v] == line[v]:
                            correct += 1
                    except:
                        pass
                    tot += 1
            cnt += 1
    return correct/tot

def export_res(pattern_expressions, dirty_path):
    res_df = pd.read_csv(dirty_path)
    for i in range(len(res_df)):
        for v in pattern_expressions[i]:
            res_df.iloc[i, list(res_df.columns).index(v)] = pattern_expressions[i][v]
    res_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Repaired_res/horizon/" + task_name[:-1] +"/repaired_" + task_name + dirty_path[-25:-4] + ".csv"
    res_df.to_csv(res_path, index=False)

def calF1(precision, recall):
    return 2*precision*recall/(precision + recall + 1e-10)

def GeneratePatternPreservingRepairs(dirty_path, constraints_path):
    g = BuildFDPatternGraph(dirty_path, constraints_path)
    ComputePatternQulity(g)
    order, tar, scc, G = BuildSCCGraghAndSort(constraints_path)
    OrderedFDs = OrderFDs(constraints_path, order, tar, scc, G)
    pattern_expressions = []
    with open(dirty_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, restval='nan')
        data = pd.read_csv(dirty_path)
        data = data.fillna("nan")
        data = data.astype(str)
        for i in range(len(data)):
        # for row in reader:
            Rtable = {}
            for v in g.vertList:
                if g.vertList[v].type == 0:
                    content = data.loc[i, g.vertList[v].attr]
                    Rtable.update({g.vertList[v].attr: content})
            for j in range(0, len(OrderedFDs)):
                if OrderedFDs[j].left not in Rtable.keys():
                    Rtable.update({OrderedFDs[j].left: data.loc[i, OrderedFDs[j].left]})
                Lval = Rtable[OrderedFDs[j].left]
                if OrderedFDs[j].right in Rtable:
                    continue
                if Lval == '':
                    Lval = 'nan'
                maxedge = -1
                for v in g.vertList[Lval].getConnections():
                    if v.attr == OrderedFDs[j].right and g.vertList[Lval].connectedQLT[v] > maxedge:
                        maxedge = g.vertList[Lval].connectedQLT[v]
                        maxp = v.id
                if maxedge != -1:
                    Rtable.update({OrderedFDs[j].right: maxp})
                if PERFECTED:
                    if (i, list(clean_df.columns).index(OrderedFDs[j].right)) not in gt_wrong_cells:
                        Rtable.update({OrderedFDs[j].right: data.loc[i, OrderedFDs[j].right]})
                    if (i, list(clean_df.columns).index(OrderedFDs[j].left)) not in gt_wrong_cells:
                        Rtable.update({OrderedFDs[j].left: data.loc[i, OrderedFDs[j].left]})
            pattern_expressions.append(Rtable)
    return pattern_expressions

def dirty_cells(dirty_file, clean_file):
    dirty_c = []
    for i in range(len(clean_file)):
        for j in range(len(clean_file.columns)):
            if dirty_file.iloc[i, j] != clean_file.iloc[i, j]:
                dirty_c.append((i, j))
    return dirty_c

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

# dirty_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/data with dc_rules/hospital/noise/hospital-inner_outer_error-01.csv"
# rule_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/data with dc_rules/hospital/dc_rules-validate-fd-horizon.txt"
# clean_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/data with dc_rules/hospital/clean.csv"
# task_name = "hospital1"
# ONLYED = 0
# PERFECTED = 0

start_time = time.time()
dirty_df = pd.read_csv(dirty_path).astype(str)
clean_df = pd.read_csv(clean_path).astype(str)
dirty_df = dirty_df.fillna("nan")
clean_df = clean_df.fillna("nan")
dirty_c = dirty_cells(dirty_df, clean_df)
gt_wrong_cells = []
for i in range(len(clean_df)):
    for j in range(len(clean_df.columns)):
        if clean_df.iloc[i, j] != dirty_df.iloc[i, j]:
            gt_wrong_cells.append((i, j))

pattern_expressions = GeneratePatternPreservingRepairs(dirty_path, rule_path)
end_time = time.time()
# print(pattern_expressions)
det_prec, det_rec = calDetPrecRec(pattern_expressions, dirty_path, clean_path)

rep_precision = calRepPrec(pattern_expressions, dirty_path, clean_path)
rep_recall = calRepRec(pattern_expressions, dirty_path, clean_path)
rep_f1 = calF1(rep_precision, rep_recall)

if True:
    if PERFECTED:
        out_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Exp_result/horizon/" + task_name[:-1] +"/perfectED+EC_" + task_name + check_string(dirty_path) + ".txt"
        res_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Repaired_res/horizon/" + task_name[:-1] +"/perfect_repaired_" + task_name + check_string(dirty_path) + ".csv"
        f = open(out_path, 'w')
        sys.stdout = f
    else:
        out_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Exp_result/horizon/" + task_name[:-1] +"/onlyED_" + task_name + check_string(dirty_path) + ".txt"
        f = open(out_path, 'w')
        sys.stdout = f
        print(det_prec)
        print(det_rec)
        print(calF1(det_prec, det_rec))
        print(end_time-start_time)
        f.close()

        out_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Exp_result/horizon/" + task_name[:-1] +"/oriED+EC_" + task_name + check_string(dirty_path) + ".txt"
        f = open(out_path, 'w')
        sys.stdout = f
        res_path = "/data/nw/DC_ED/References_inner_and_outer/DATASET/Repaired_res/horizon/" + task_name[:-1] +"/repaired_" + task_name + check_string(dirty_path) + ".csv"
    print(rep_precision)
    print(rep_recall)
    print(rep_f1)
    print(end_time-start_time)
    f.close()

    res_df = pd.read_csv(dirty_path)
    for i in range(len(res_df)):
        for v in pattern_expressions[i]:
            res_df.loc[i, v] = pattern_expressions[i][v]
    res_df.to_csv(res_path, index=False)