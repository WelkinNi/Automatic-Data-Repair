import os
import pandas as pd
import numpy as np
import csv
import copy
from mvc import  read_graph,min_vertex_cover,read_graph_dc
from tes import greedy_min_vertex_cover,greedy_min_vertex_cover_dc
from DC_Rules import DCRule
import time
import sys

# from queue import PriorityQueue
# from graph import MinimumVertexCoverSolver
class bigdansing():
    def __init__(self):
        self.wrong = []
        self.rule = []
        self.blocked_list = [[]]
        self.data = []
        self.data_cl = []
        self.input_data = []
        self.dicc = {}
        self.visdic = {}
        self.edgedic = {}
        self.diccop = {}
        self.maypair = [[]]
        self.scdic = {}
        self.dic = {}
        self.scodic = {}
        self.clean_right = 0
        self.all_clean = 0
        self.clean_right_pre = 0
        self.Rules = []
        self.exps = []
        self.mvc = []
        self.sorts = []
        self.vio = []
        self.cou = 0
    '''
        scope操作符：从数据集中删除不相关的数据单元
        Parameters
        ----------
        sco :
            要从数据集中取出的列名，是一个list
        Returns
        -------
        data :
            返回从数据集中取回的data
    '''
    def scope(self, sco):
        df = pd.read_csv('dirty.csv', sep=',', header=0)
        data = np.array(df[sco]).tolist()
        for i in data:
            for j in i:
                j = str(j)
        return data

    def scope1(self, sco):
        # #数据集beers预处理
        for i in range(len(sco)):
            if (sco[i] == 'brewery_name'):
                sco[i] = 'brewery-name'
        # print(sco)
        df = pd.read_csv('clean.csv', sep=',', header=0)
        data = np.array(df[sco]).tolist()
        return data

    '''
        block操作符：将共享了相同的blocking key的数据进行分组
        Parameters
        ----------
        data :
            数据集
        blo :
            blocking key,block操作符根据blo进行分组
        Returns
        -------
        blocked_list :
            返回根据blo分好块的data,为一个二层list，list[i]表示第i组blo一样的tuple
            example:
                [[0, 1],[2, 3]]中表示有2组值，其中0，1的blo值相同且2，3的blo值相同
    '''

    def block(self, data, blo):
        blodic = {}
        blodiccou = 0
        for i in range(len(data)):
            if (pd.isna(data[i][blo])):
                if (blodic.__contains__("nan")):
                    self.blocked_list[blodic["nan"]].append(i)
                else:
                    blodic.setdefault("nan", blodiccou)
                    self.blocked_list.append([])
                    self.blocked_list[blodic["nan"]].append(i)
                    blodiccou += 1
                    # continue
            elif (blodic.__contains__(data[i][blo])):
                # print(lii[blodic[li[i][blo]]])
                self.blocked_list[blodic[data[i][blo]]].append(i)
            else:
                blodic.setdefault(data[i][blo], blodiccou)
                self.blocked_list.append([])
                self.blocked_list[blodic[data[i][blo]]].append(i)
                blodiccou += 1
        # print(blocked_list)
        return self.blocked_list

    '''
        iterate操作符：根据block操作中分完组的各个组生成其潜在可能违规的组,返回pair，生成候选冲突
        Parameters
        ----------
        data :
            数据集
        blocked_list :
            存放分好组的list列表
        Returns
        -------
        pair :
            返回根据blocked_list生成的候选冲突对,是一个三层的列表，pair[i]表示中blo中值相等，pair[i][j]表示潜在的违规对
            example :
                [1, 2]表示1，2条数据为潜在的冲突对
    '''

    def iterate(self, data, blocked_list):
        pair = [[]]
        for i in range(len(blocked_list)):
            for j in range(len(blocked_list[i])):
                for k in range(j + 1, len(blocked_list[i])):
                    pair[i].append([blocked_list[i][j], blocked_list[i][k]])
            pair.append([])
        # print(pair)
        return pair

    '''
        generate：根据一个列表和一个字典生成一个表达式
        Parameters
        ----------
        newtemdic :
            列表，表示这条中规则要用到的列
        temdic :
            字典，指向每个列名对应违反的操作符
        Returns
        -------
        bds :
            返回生成的判断表达式
            example :
                ['str(li[l][scdic["ounces"]])!=str(li[r][scdic["ounces"]]) and int(li[l][scdic["brewery_id"]])>int(li[r][scdic["brewery_id"]])'
    '''

    def generate(self, index):
        zhuanhuan = ["==", "!=", "<", ">", "<=", ">="]
        biaodashi = []
        bds = ""
        for predicate in self.Rules[index].predicates:
            if (predicate.opt == '<>'):
                biaodashi.append("str(data[l][self.scdic[\"" + str(predicate.components[0]) + "\"]])" + "!=" + "str(data[r][self.scdic[\"" + str(predicate.components[1]) + "\"]])")
            elif(predicate.opt == '='):
                biaodashi.append("str(data[l][self.scdic[\"" + str(
                    predicate.components[0]) + "\"]])" + "==" + "str(data[r][self.scdic[\"" + str(
                    predicate.components[1]) + "\"]])")
            else:
                biaodashi.append("str(data[l][self.scdic[\"" + str(predicate.components[0]) + "\"]])" + predicate.opt + "str(data[r][self.scdic[\"" + str(predicate.components[1]) + "\"]])")

        for i in range(len(biaodashi)):
            if (i == 0):
                bds = biaodashi[i]
            else:
                bds = bds + " and " + biaodashi[i]
        return bds

    '''
        detect：根据maypair和生成的表达式从潜在的违规maypair中生成真正的违规list：vio
        Parameters
        ----------
        maypair :
            iterate中生成的潜在违规对
        data :
            数据集
        Returns
        -------
        vio :
            返回生成的违规超边
            example :
                <class 'list'>: [0, 1, (0, 1), (50, 1)]
                    其中vio[i]即为第i条超边
                    其中第1个数字，0表示违反了第0条规则，之后1个数字和2个元组指示了一组违规
                    3个数字中的第一个表示他们的操作符，如1表示"!=",后两个元组，表示违规的2个cell
                    如(0, 1)表示第0行第1列的cell
    '''
    def ocjoin(self, index, blocked_list, data):
        # print("------------------------------------")
        finaltuples = []
        nbparts = 10
        conds = []
        K = [[]]
        Kcou = 0
        flagg = 0
        #partatt = conds[0]
        for predicate in self.Rules[index].predicates:
            if (predicate.opt != '=' and predicate.opt != '<>' ):
                conds.append([predicate.components[0], predicate.opt])
        for i in range(len(blocked_list)):
            K = [[]]
            k = 0
            Kcou = 0
            endk = 0
            while (k < len(blocked_list[i])):
                try:
                    K[Kcou] = blocked_list[i][k :k + nbparts]
                except:
                    K[Kcou] = blocked_list[i][k : len(blocked_list[i])-1]
                k += nbparts
                Kcou += 1
                K.append([])
            for kj in range(len(K)):
                for cond in conds:
                    condindex = self.scdic[cond[0]]
                    for Kin in range(len(K[kj])):
                        try:
                            data[K[kj][Kin]][condindex] = int(data[K[kj][Kin]][condindex])
                        except:
                            data[K[kj][Kin]][condindex] = 0
                        self.sorts[condindex].append([data[K[kj][Kin]][condindex], Kin])
                    for sortsindex in range(len(self.sorts[condindex])):
                        try:
                            self.sorts[condindex][sortsindex][0] = int(self.sorts[condindex][sortsindex][0])
                        except:
                            self.sorts[condindex][sortsindex][0] = 0
                    self.sorts[condindex] = sorted(self.sorts[condindex], key=lambda x: x[0])
                for kl in range(kj+1, len(K)):
                    # print(self.scdic[conds[0][0]])
                    kjmax = float(self.sorts[self.scdic[conds[0][0]]][len(self.sorts[self.scdic[conds[0][0]]]) - 1][0])
                    kjmin = float(self.sorts[self.scdic[conds[0][0]]][0][0])
                    klmax =  - np.inf
                    klmin = np.inf
                    for cond in conds:
                        condindex = self.scdic[cond[0]]
                        for Kin in range(len(K[kl])):
                            try:
                                float(data[K[kl][Kin]][condindex])
                            except:
                                data[K[kl][Kin]][condindex] = 0
                    for Kin in range(len(K[kl])):
                        # print(data[K[kl][Kin]][self.scdic[conds[0][0]]])
                        klmax = max(float(klmax) , float(data[K[kl][Kin]][self.scdic[conds[0][0]]]))
                        klmin = min(float(klmin) , float(data[K[kl][Kin]][self.scdic[conds[0][0]]]))

                    if ((kjmax < klmin) or (kjmin > klmax)):
                        continue
                    else:
                        tuples = []
                        for cond in conds:
                            temptuples = []
                            bds = "float(sortsindex[0]) " + str(cond[1]) + " float(data[klindex][self.scdic[cond[0]]])"
                            for klindex in K[kl]:
                                for sortsindex in self.sorts[self.scdic[cond[0]]]:
                                    flag = 0
                                    if(flag == 1):
                                        temptuples.append((K[kj][sortsindex[1]] , klindex))
                                    if(eval(bds)):
                                        try:
                                            temptuples.append((K[kj][sortsindex[1]] , klindex))
                                        except:
                                            print("kj:",kj,"sortsindex[1]",sortsindex[1])
                                        flag = 1
                            if (tuples == []):
                                tuples = temptuples

                            else:
                                for pair1 in tuples:
                                    if (pair1 in temptuples):
                                        flagg = 1
                                    else:
                                        tuples.remove(pair1)
                                    if (flagg == 1):
                                        flagg = 0
                                        break
                        finaltuples.extend(tuples)
        anotemdic = {"=": 0, "<>": 1, "<": 2, ">": 3, "<=": 4, ">=": 5}
        for pair in finaltuples:


            biaodashi = self.generate(index)
            l = pair[0]
            r = pair[1]
            # print("rule:", index, "l:", l, "r:", r)
            if (eval(biaodashi)):
                self.vio.append([index])
                for predicate in self.Rules[index].predicates:
                    self.vio[self.cou].append(anotemdic[predicate.opt])
                    self.vio[self.cou].append((l, self.scdic[predicate.components[0]]))
                    self.vio[self.cou].append((r, self.scdic[predicate.components[1]]))
                self.cou += 1

    def detect(self, maypair, data):
        newdic = list(self.dic)
        # print("newdic:", newdic)
        newscodic = list(self.scodic.values())
        # print("newscodic:", newscodic)
        zhuanhuan = ["==", "!=", "<", ">", "<=", ">="]
        # 第一层循环表示现在是在使用第i个规则，第二层循环是分好的每个block块，第3层循环中的k为潜在违规对
        for i in range(len(maypair) - 1):
            temdic = {}
            anotemdic = {"=": 0, "<>": 1, "<": 2, ">": 3, "<=": 4, ">=": 5}
            biaodashi = self.generate(i)
            for j in range(len(maypair[i])):
                for k in maypair[i][j]:
                    l = k[0]
                    r = k[1]
                    try:
                        eval(biaodashi)
                    except:
                        for predicate in self.Rules[i].predicates:
                            if (predicate.opt != '=' and predicate.opt != '<>'):
                                try:
                                    float(data[l][predicate.components[0]])
                                except:
                                    data[l][predicate.components[0]] = 0
                                try:
                                    float(data[r][predicate.components[1]])
                                except:
                                    data[r][predicate.components[1]] = 0

                    if (eval(biaodashi)):
                        self.vio.append([i])
                        for predicate in self.Rules[i].predicates:
                            self.vio[self.cou].append(anotemdic[predicate.opt])
                            self.vio[self.cou].append((l, self.scdic[predicate.components[0]]))
                            self.vio[self.cou].append((r, self.scdic[predicate.components[1]]))
                        self.cou += 1
        return self.vio

    '''
        repair：跟据data和detect中生成的vio进行修复，是holistic算法中的algorithm1
                生成超图，对超图进行mvc算法，mvc得出错误cell，之后用lookup进行frontier的寻找，并得出表达式
                最后用determination得出最终修复，无法修复的根据postprocess进行修复
        Parameters
        ----------
        vio :
            detect中生成的违规对的集合
        data :
            数据集
        Returns
        -------
        data :
            完成修复的数据
        all_clean :
            进行的全部修复的总次数
        clean_right :
            进行的修复中的修复正确的次数
        clean_right_pre :
            进行的修复中的修复正确的次数,用来计算prec
    '''

    def repair(self, data, vio):
        clean_right = 0
        clean_right_pre = 0
        all_clean = 0
        sizebefore = 0
        sizeafter = 0
        processedcell = []
        # fh = open("edges.txt", "w")
        # for i in vio:
        #     fh.write(str(str(i[0])+" "+str(i[1])+ " 1"))
        #     fh.write("\n")
        # fh.close()
        # fh = open("edges.txt", "r")
        # dataa=fh.read()
        # print("start")
        # print(dataa)
        # print("end")

        # S = min_vertex_cover(G, vio[0][0])


        # read_graph_dc算法以vio作为输入，输出一个字典
        # 字典中例如(15, 7)：[1,2,3,4,5]表示第15行第7列的cell存在于第1，2，3，4，5条超边中，
        # 这个超边的编号与vio中的编号对应


        input_data = read_graph_dc(vio)
        # print(input_data)
        dicc = input_data.copy()
        for i in dicc:
            dicc[i] = list(set(dicc[i]))
        # print("dicc:",dicc)
        for i in dicc.items():
            processedcell.append(i[0])
        sizebefore = len(processedcell)
        while (sizebefore > sizeafter):
            sizebefore = len(processedcell)
            input_data = read_graph_dc(vio)
            # dicc和diccop中放的都是第i条vio，vio中是超边
            dicc = input_data.copy()
            for i in dicc:
                dicc[i] = list(set(dicc[i]))
            diccop = copy.deepcopy(dicc)
            # 精确贪心mvc算法，输出为mvc找到的cell的list
            self.mvc = greedy_min_vertex_cover_dc(dicc, vio)
            # print("mvc:",len(mvc))
            mvcdic = copy.deepcopy(self.mvc)
            # print("dic:",diccop)
            # print("mvc:",mvc)
            while self.mvc:
                cell = self.mvc.pop()
                # print("cell:",cell)
                edges = dicc[cell]
                # r = rc(cell)
                # r.exps.clear()
                while edges:
                    edge = edges.pop()
                    index1 = vio[edge].index(cell)
                    if (index1 % 3 == 2):
                        index2 = index1 + 1
                        index0 = index1 - 1
                    if (index1 % 3 == 0):
                        index2 = index1 - 1
                        index0 = index1 - 2
                    if (index1 % 3 == 1):
                        continue
                    self.visdic.clear()
                    self.edgedic.clear()
                    # lookup算法，作用是找出cell所有相关联的边
                    exps = []
                    exps = self.lookup(cell, vio[edge][index2], vio[edge][index0], diccop, mvcdic, vio, cell)
                l = cell[0]
                rr = cell[1]
                # print("start")
                # print(exps)
                # print("end")
                truerepair = self.determination(cell, exps, data)
                self.exps.clear()
                # print("unrepair:", li[l][rr])
                # print("repair:",truerepair)
                # print("realthing:",li_cl[l][rr])
                # print("all_clean+1")


                # 计算了总的修复次数
                self.all_clean = self.all_clean + 1
                # print(all_clean)

                # 计算了总的改对的个数
                # print("truerepair:", truerepair, "data_cl:", self.data_cl[l][rr], "dataorg:", data[l][rr],"l:",l,"r:",rr)
                if ((str(truerepair) == str(self.data_cl[l][rr]))):
                    self.clean_right_pre = self.clean_right_pre + 1
                if ((str(truerepair) == str(self.data_cl[l][rr])) and (data[l][rr] != self.data_cl[l][rr])):
                    self.clean_right = self.clean_right + 1
                data[l][rr] = truerepair

            vio = self.detect(self.maypair, data)
            # print(vio)
            # input_data = read_graph(vio)
            # left_v = input_data[0]
            # right_v = input_data[1]
            # dicc = left_v.copy()
            # for i in list(right_v):
            #     if (dicc.__contains__(i)):
            #         dicc[i].extend(right_v[i])
            #         del (right_v[i])
            # dicc.update(right_v)
            # for i in dicc:
            #     dicc[i] = list(set(dicc[i]))
            # mvc = greedy_min_vertex_cover(dicc)
            if (len(list(dicc)) == 0):
                return data, self.all_clean, self.clean_right
            for i in dicc.items():
                processedcell.append(i[0])
            sizeafter = len(processedcell)
        self.all_clean, self.clean_right, self.clean_right_pre = self.postprocess(self.mvc, dicc, data, self.all_clean, self.clean_right, self.clean_right_pre)
        return data, self.all_clean, self.clean_right, self.clean_right_pre

    '''
        postprocess：跟对repair的内循环无法修复的mvc中的单元进行粗修复
        Parameters
        ----------
        mvc :
            mvc算法得出的cell
        dicc :
            是一个字典，存放了整个超图
            example:
                {(1, 1): [2, 3, 4]}表示第1行第1列的cell存在于2，3，4三条超边中
        data :
            数据集
        all_clean :
            进行的全部修复的总次数
        clean_righ :
            进行的修复中的修复正确的次数
        Returns
        -------
        all_clean :
            进行的全部修复的总次数
        clean_righ :
            进行的修复中的修复正确的次数
    '''

    def postprocess(self, mvc, dicc, data, all_clean, clean_right, clean_right_pre):
        while mvc:
            cell = mvc.pop()
            # print("cell:",cell)
            edges = dicc[cell]
            edge = edges.pop()
            index1 = self.vio[edge].index(cell)
            if (index1 % 3 == 2):
                index2 = index1 + 1
                index0 = index1 - 1
            if (index1 % 3 == 0):
                index2 = index1 - 1
                index0 = index1 - 2
            if (index1 % 3 == 1):
                continue
            l1 = cell[0]
            r1 = cell[1]
            l2 = self.vio[edge][index2][0]
            r2 = self.vio[edge][index2][1]
            if (self.vio[edge][index0] == 1):
                truerepair = self.data[l2][r2]
            if (self.vio[edge][index0] == 2):
                truerepair = self.data[l2][r2] - 1
            if (self.vio[edge][index0] == 3):
                truerepair = self.data[l2][r2] + 1
            if (self.vio[edge][index0] == 4):
                truerepair = self.data[l2][r2]
            if (self.vio[edge][index0] == 5):
                truerepair = self.data[l2][r2]
            all_clean += 1
            if (str(truerepair) == str(self.data_cl[l1][r1])):
                clean_right_pre = clean_right_pre + 1
            if ((str(truerepair) == str(self.data_cl[l1][r1])) and (self.data[l1][r1] != self.data_cl[l1][r1])):
                clean_right = clean_right + 1
            self.data[l1][r1] = truerepair
        return all_clean, clean_right, clean_right_pre

    # 参数为cell：违规的单元，edge：关联的边，oper：cell和edge违反的操作符，rc即repaircontext，存放表达式
    # diccop是一个字典字典中例如157：[1,2,3,4,5]表示第15行第7列的cell存在于第1，2，3，4，5条超边中，
    # mvcdic即为mvc得出的list，vio即为超边与上述的vio相同
    # 返回的是rc，rc中存放了所有的潜在的修复表达式
    '''
        lookup ：找到违规单元cell所有相关联的边，生成表达式
        Parameters
        ----------
        cell :
            违规的单元
        edge ;
            关联的边
        oper ;
            cell和edge违反的操作符
        rc :
            repaircontext,修复上下文，存放了表达式
        diccop :
            是一个字典，存放了整个超图
            example:
                {(1, 1): [2, 3, 4]}表示第1行第1列的cell存在于2，3，4三条超边中
        mvcdic :
            判断该值是否存在于mvc中
        vio :
            detect中生成的违规对的集合
        Returns
        -------
        rc :
            生成的修复上下文，里面存放了表达式
    '''

    def lookup(self, cell, edge, oper, diccop, mvcdic, vio, firstcell):
        # cell,edge 意为cell被修复为edge
        # print("lookup:",diccop)
        # print("cell:",cell,"edge:",edge)
        if (firstcell != cell):
            return []
        self.exps.extend([[cell, edge, oper]])
        # front中存放的为若干条超边，其中的值是第i条vio数据
        front = []
        # print(diccop[cell])
        try:
            front.extend(diccop[cell])
        except:
            pass
        # print("front:",front)
        for i in front:
            index1 = vio[i].index(cell)
            if (index1 % 3 == 2):
                index2 = index1 + 1
                index0 = index1 - 1
            if (index1 % 3 == 0):
                index2 = index1 - 1
                index0 = index1 - 2
            if (mvcdic.__contains__(vio[i][index2]) and vio[i][index2] != cell):
                continue
            if (self.visdic.__contains__(vio[i][index2])):
                continue
            # if (edgedic.__contains__(i)):
            #     continue
            # edgedic[i] = 1
            self.visdic[vio[i][index2]] = 1
            try:
                edges = diccop[vio[i][index2]]
            except:
                edges = []
                continue
            # print("edges:",edges)
            for j in edges:
                index11 = vio[j].index(vio[i][index2])
                if (index11 % 3 == 2):
                    index22 = index11 + 1
                    index00 = index11 - 1
                if (index11 % 3 == 0):
                    index22 = index11 - 1
                    index00 = index11 - 2
                # print("1:",vio[i][index2],"index11:",index11,"index22:",index22,"2:",vio[i][index22])
                if (mvcdic.__contains__(vio[j][index22])):
                    continue
                if (self.visdic.__contains__(vio[j][index22])):
                    continue
                # if (edgedic.__contains__(j)):
                #     continue
                self.visdic[vio[j][index22]] = 1
                # edgedic[j] = 1
                self.exps.extend(self.lookup(vio[i][index2], vio[j][index22], vio[j][index00], diccop, mvcdic, vio, cell))
        return self.exps


    # determination
    # 对rc中的表达式寻找最合适的修复，返回值为对于cell最终的修复
    '''
        determination ：对rc中的表达式寻找最合适的修复，返回值为对于cell最终的修复
        Parameters
        ----------
        cell :
            违规的单元
        exps :
            修复表达式
        data :
            数据集
        Returns
        -------
        finalthing or ll[0] :
            均为对于cell最终的修复
    '''

    def determination(self, cell, exps, data):
        zhuanhuan = ["==", "!=", "<", ">", "<=", ">="]
        repairdic = {}
        repairdic.clear()
        temp11 = exps[0][0][0]
        temp22 = exps[0][0][1]
        # H = np.eye(len(rc.exps) + 1)
        # f = np.array([-2 * li[temp11][temp22]])
        # L = np.array([])
        # k = np.array([])
        cou = 0
        max = -np.inf
        min = np.inf
        finalthing = -2
        for i in exps:

            temp1 = i[1][0]
            temp2 = i[1][1]

            # print(i[2])
            if (zhuanhuan[i[2]] == "=="):
                continue
            if (zhuanhuan[i[2]] == "!="):
                if (repairdic.__contains__(data[temp1][temp2])):
                    # print(1)
                    repairdic[data[temp1][temp2]] = repairdic[data[temp1][temp2]] + 1
                else:
                    repairdic[data[temp1][temp2]] = 1
            else:
                # f=np.append(f, -2*data[temp1][temp2])
                if (zhuanhuan[i[2]] == "<" or zhuanhuan[i[2]] == "<="):
                    if (max < float(data[temp1][temp2])):
                        max = float(data[temp1][temp2])
                    finalthing = max + 1
                    # L=np.eye(len(rc.exps) - 1)
                    # L=L-2
                    # L=np.insert(L,0,1,axis = 1)
                else:
                    if (min > float(data[temp1][temp2]) and data[temp1][temp2] != 0):
                        min = float(data[temp1][temp2])
                    finalthing = min - 1
                    #     L = np.eye(len(rc.exps) - 1)
                    #     L = np.insert(L, 0, -1, axis=1)
                    # k=np.append(k,0)
        # print(H, f, L, k)
        # x = solve_qp(H, f, L, k,solver="osqp")
        if (finalthing != -2):
            return finalthing
        sorted(repairdic.items(), key=lambda x: x[1])
        ll = list(repairdic)
        # print(repairdic)
        try:
            return ll[0]
        except:
            return 0

    def run(self, file_path, path):
        os.chdir(path)
        #获取列名
        df = pd.read_csv('dirty.csv', header=None)
        schema = np.array(df).tolist()[0]
        f = open("dc_rules.txt", "r")
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            file = f.read()
        rules = file.split('\n')

        # 要进行scope的
        sco = []

        diccou = 0
        flag = 0

        # 记录等号数量和非等号规则的数量，如果规则只含有等号，则不做任何处理
        equal_num = 0
        other_num = 0

        # 对于dc规则的处理
        # 规定规则为此类型格式∀t0∈clean.csv,t1∈clean.csv:¬[t0.src=t1.src∧t0.act_arr_time=t1.act_arr_time]

        for rule in rules:
            equal_num = 0
            other_num = 0
            DR = DCRule(rule, schema)
            for predicate in DR.predicates:
                sco.append(predicate.components[0])
                sco.append(predicate.components[1])
                if (predicate.opt == '='):
                    equal_num += 1
                else:
                    other_num += 1
            if (equal_num == 0):
                DR_copy = copy.deepcopy(DR)
                DR_copy.att_copy(DR)
                self.Rules.append(DR_copy)
            elif (other_num == 0):
                DR_copy = copy.deepcopy(DR)
                DR_copy.att_copy(DR)
                self.Rules.append(DR_copy)
            else:
                DR_copy = copy.deepcopy(DR)
                DR_copy.att_copy(DR)
                self.Rules.append(DR_copy)
            # # print(f'epoch {epoch + 1}, loss {l:f}')
            # Rulei = Rule()
            # equal_num = 0
            # other_num = 0
            # i = i.split("¬")
            # # 分割出i[0]没用，i[1]示例为[t0.state=t1.state∧t0.ounces≠t1.ounces∧t0.brewery_id>t1.brewery_id]
            # sum_rule = i[1][1:-1]
            # rules = sum_rule.split("∧")
            # for rule in rules:
            #
            #     opers = ['≠', '<', '>', '≤', '≥']
            #
            #     if ('=' in rule):
            #         equal_num += 1
            #         temp = rule.split('=')
            #         temp0 = temp[0].split('.')
            #         sco.append(temp0[1])
            #         Rulei.attribute[0].append(temp0[1])
            #         temp1 = temp[1].split('.')
            #         sco.append(temp1[1])
            #         Rulei.attribute[0].append(temp1[1])
            #     for oper in range(len(opers)):
            #         if (opers[oper] in rule):
            #             other_num += 1
            #             temp = rule.split(opers[oper])
            #             temp0 = temp[0].split('.')
            #             sco.append(temp0[1])
            #             Rulei.attribute[oper + 1].append(temp0[1])
            #             temp1 = temp[1].split('.')
            #             sco.append(temp1[1])
            #             Rulei.attribute[oper + 1].append(temp1[1])
            # Rulei.att_set()

        scdiccou = 0
        for i in range(len(sco)):
            if (sco[i] == 'city '):
                sco[i] = 'city'
        sco = list(set(sco))
        for i in sco:
            self.sorts.append([])
            self.scdic.setdefault(i, scdiccou)
            scdiccou += 1

        # print("scdic:",scdic)
        data = self.scope(sco)
        yuanshi = []
        yuanshi = self.scope(sco)
        # all_wrong的计算
        all_wrong = 0
        self.data_cl = self.scope1(sco)
        # print(li_cl)
        df = pd.read_csv('dirty.csv', sep=',', header=0)
        data_wrong = np.array(df).tolist()
        df = pd.read_csv('clean.csv', sep=',', header=0)
        data_clean = np.array(df).tolist()
        # 对比scope中找出的数据集和clean中的数据集，计算出全部错误cell的个数
        for i in range(len(data_wrong)):
            for j in range(len(data_wrong[i])):
                try:
                    float(data_wrong[i][j])
                    float(data_clean[i][j])
                    if (float(data_wrong[i][j]) != float(data_clean[i][j])):
                        if (pd.isna(data_wrong[i][j]) and pd.isna(data_clean[i][j])):
                            continue
                        all_wrong += 1
                        print("Wrong:",data_wrong[i][j],"clean:",data_clean[i][j])
                except:
                    if (str(data_wrong[i][j]) != str(data_clean[i][j])):
                        if (pd.isna(data_wrong[i][j]) and pd.isna(data_clean[i][j])):
                            continue
                        all_wrong += 1
                        print("Wrong:", data_wrong[i][j], "clean:", data_clean[i][j])
        print("all_wrong:", all_wrong)
        vio = []
        # 进行block和iterate操作
        #sadklj = [i[0] for i in data]
        for i in range(len(self.Rules)):
            icou = 0
            temflag = 0
            i_equalnum = 0
            i_othernum = 0
            i_ocjoinnum = 0
            equal_components = []
            for predicate in self.Rules[i].predicates:
                if (predicate.opt == '=' and predicate.property[0] == "attribute" and predicate.property[
                    1] == "attribute"):
                    equal_components.append(predicate.components[0])
            for predicate in self.Rules[i].predicates:
                if (predicate.opt == '='):
                    i_equalnum += 1
                else:
                    i_othernum += 1
                    if (predicate.opt != '<>'):
                        i_ocjoinnum += 1
            if (i_othernum == 0):
                for j in range(len(sco)):
                    for k in range(len(equal_components)):
                        if (equal_components[k] == sco[j]):
                            self.blocked_list = []
                            self.block(data, j)
                            if ((len(equal_components) > 1) and flag == 1):
                                newli = self.iterate(data, self.blocked_list)
                                for pairlist in self.maypair[icou]:
                                    for pair1 in pairlist:
                                        for pairlist2 in newli:
                                            if (pair1 in pairlist2):
                                                flagg = 1
                                            if (flagg == 1):
                                                flagg = 0
                                                break
                                            if (pairlist2 == newli[-1]):
                                                pairlist.remove(pair1)
                                                break
                                    pass
                            else:
                                flag = 1
                                self.maypair[icou].extend(self.iterate(data, self.blocked_list))
                icou += 1
                self.maypair.append([])
            elif (i_equalnum == 0 and len(self.Rules[i].variable) > 1):
                if (i_ocjoinnum == 0):
                    blocked_list = [[]]
                    for j in range(len(data)):
                        blocked_list[0].append(j)
                    self.maypair[icou].extend(self.iterate(data, blocked_list))
                    icou += 1
                    self.maypair.append([])
                else:
                    blocked_list = [[]]
                    for j in range(len(data)):
                        blocked_list[0].append(j)
                    self.ocjoin(i, self.blocked_list, data)
            elif (i_equalnum == 0 and len(self.Rules[i].variable) == 1):
                for j in range(len(data)):
                    self.maypair[icou].append([j, j])
                icou += 1
                self.maypair.append([])
            elif (i_ocjoinnum == 0):
                for j in range(len(sco)):
                    for k in range(len(equal_components)):
                        if (equal_components[k] == sco[j]):
                            self.blocked_list = []
                            self.block(data, j)
                            if ((len(equal_components) > 1) and flag == 1):
                                newli = self.iterate(data, self.blocked_list)
                                for pairlist in self.maypair[icou]:
                                    for pair1 in pairlist:
                                        for pairlist2 in newli:
                                            if (pair1 in pairlist2):
                                                flagg = 1
                                            if (flagg == 1):
                                                flagg = 0
                                                break
                                            if (pairlist2 == newli[-1]):
                                                pairlist.remove(pair1)
                                                break
                                    pass
                            else:
                                flag = 1
                                self.maypair[icou].extend(self.iterate(data, self.blocked_list))
                icou += 1
                self.maypair.append([])
            elif (i_ocjoinnum != 0):
                for j in range(len(sco)):
                    for k in range(len(equal_components)):
                        if (equal_components[k] == sco[j]):
                            self.blocked_list = []
                            self.block(data, j)
                            self.ocjoin(i, self.blocked_list, data)
            for i in range(len(self.sorts)):
                self.sorts[i].clear()

        viocount = 0
        self.vio = self.detect(self.maypair, data)
        liclean, all_clean, clean_right, clean_right_pre = self.repair(data, self.vio)
        print("all_clean:", all_clean)
        print("clean_right:", clean_right)
        print("clean_right_pre:", clean_right_pre)
        prec = clean_right_pre * 1.0 / all_clean
        rec = clean_right * 1.0 / all_wrong
        if (prec == 0 and rec == 0):
            f1 = 0
        else:
            f1 = 2 * prec * rec / (prec + rec)
        print("prec=", prec)
        print("rec=", rec)
        print("f1=", f1)

class Rule():
    #等于，不等，小于，大于，小于等于，大于等于
    attribute = [[],[],[],[],[],[]]
    def __init__(self):
        for i in self.attribute:
            i.clear()
    def att_set(self):
        for i in range(6):
            self.attribute[i]=list(set(self.attribute[i]))

    def att_copy(self,othe):
        self.attribute = copy.deepcopy(othe)

pd.set_option('display.max_columns', None)
# 不限制最大显示行数
pd.set_option('display.max_rows', None)
#
#
if __name__ == "__main__":
    time_start = time.time()
    path = "/data/nw/DC_ED/datasets/flights"
    file_path = ("/data/nw/DC_ED/datasets/flights/dc_rules.txt")
    bd = bigdansing()

    bd.run(file_path, path)
    time_end = time.time()
    print('time cost',time_end - time_start,'s')