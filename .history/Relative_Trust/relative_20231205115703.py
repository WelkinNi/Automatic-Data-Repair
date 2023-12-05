import pandas as pd
import numpy as np
import gc
import sys
import copy
import queue
import random
import logging
import datetime
import argparse
import time
from line_profiler import LineProfiler
from sklearn.covariance import MinCovDet
from tqdm import tqdm
visited_nodes = {}


class Node():
    def __init__(self, key, state):
        self.idx = key
        self.child = []
        self.state = copy.deepcopy(state)
    
    def add_child(self, key):
        self.child.append(key)


class Tree():
    def __init__(self, root_state):
        self.node_dict = {}
        self.node_num = 1
        key = len(self.node_dict)
        new_node = Node(key, root_state)
        self.node_dict[key] = new_node
    
    def add_node(self, parent, state):
        key = len(self.node_dict)
        new_node = Node(key, state)
        self.node_dict[key] = new_node
        self.node_dict[parent].add_child(key)
        self.node_num = self.node_num + 1
        return key


class Vertex():
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}
        self.edge_list = []

    def add_neighbor(self, nbr, edg):
        if nbr in self.connectedTo:
            self.connectedTo[nbr].append(edg)
        else:
            self.connectedTo[nbr] = []
            self.connectedTo[nbr].append(edg)
        self.edge_list.append(edg)

    def __str__(self):
        return str(self.id) + 'connectedTo' + str([k for k, _ in self.connectedTo.items()])

    def getConnections(self):
        return list(self.connectedTo.keys())

    def getId(self):
        return self.id


class Edge():
    def __init__(self, key, head, tail, fd):
        self.id = key
        self.verList = []
        self.verList.append(head)
        self.verList.append(tail)
        self.fd = fd
    
    def getVertex(self):
        return list(self.verList)
    
    def getId(self):
        return self.id
    
    def getFd(self):
        return self.fd


class Graph():
    def __init__(self):
        self.vertList = {}
        self.edgeList = {}
        self.numVertices = 0
        self.numEdge = 0

    def addVertex(self, key):
        if key not in self.vertList:
            self.numVertices = self.numVertices + 1
            newVertex = Vertex(key)
            self.vertList[key] = newVertex
            return newVertex

    def addEdge(self, h, t, fd):
        self.addVertex(h)
        self.addVertex(t)
        e = len(self.edgeList)
        for edge in self.vertList[h].edge_list:
            if h in self.edgeList[edge].verList and t in self.edgeList[edge].verList and \
                fd == self.edgeList[edge].fd:
                return None
        if e not in self.edgeList:
            self.numEdge = self.numEdge + 1
            newEdge = Edge(e, h, t, fd)
            self.edgeList[e] = newEdge
            self.vertList[h].add_neighbor(self.vertList[t], e)
            self.vertList[t].add_neighbor(self.vertList[h], e)
        
    def getVertices(self):
        return self.vertList.keys()
    
    def getEdges(self):
        return list(self.edgeList.keys())

    def __iter__(self):
        return  iter(self.vertList.values())
        
    def __contains__(self, n):
        return  n in self.vertList

def print_tree(tree):
    que = queue.Queue(maxsize=0)
    que.put(0)
    layer = 0
    while not que.empty():
        childs = []
        print("LAYER: {layer}".format(layer=layer))
        while not que.empty():
            node = que.get()
            childs.extend(tree.node_dict[node].child)
            print(tree.node_dict[node].idx)
        for c in childs:
            que.put(c)
        layer = layer + 1
    

class RelaTrust:
    def __init__(self, csv_d, csv_c, fd_path):
        self._resolve_fd(fd_path)
        self.cur_fd = copy.deepcopy(self.ori_fd)
        self.dirty_csv = pd.read_csv(csv_d).astype(str)
        self.dirty_csv.fillna("null")
        self.rep_csv = copy.deepcopy(self.dirty_csv)
        self.schema = list(self.dirty_csv.columns)
        self.clean_csv = pd.read_csv(csv_c).astype(str)
        self.clean_csv.fillna("null")
        self.rep_cells = []
        self.wrong_cells = []
        self.out_csv = csv_d[:-4] + "-relatrust_cleaned.csv"
        self.graph = Graph()
        self.tau = 0.01*len(self.schema)*len(self.dirty_csv)
        for i in range(len(self.dirty_csv)):
            for j in range(len(self.dirty_csv.columns)):
                if self.dirty_csv[self.dirty_csv.columns[j]][i] != self.clean_csv[self.dirty_csv.columns[j]][i]:
                    self.wrong_cells.append((i, j))

    def _resolve_fd(self, fd_path):
        ori_f = open(fd_path)
        lines = ori_f.readlines()
        self.ori_fd = {}
        fd_num = 0
        for line in lines:
            fd_this = []
            line = line.split("⇒", 1)
            left, right = [i.strip() for i in line[0].strip().split(",")], \
                [i.strip() for i in line[1].strip().split(",")]
            fd_this.append(left)
            fd_this.append(right)
            self.ori_fd[fd_num] = fd_this
            fd_num = fd_num + 1

    def repair_data(self):
        logging.debug("Building conflict graph using updated constraints")
        self.graph = self._build_graph(self.cur_fd)
        logging.debug("Calculating MVC of the graph")
        c_2opt = self.mvc(self.graph)
        while c_2opt:
            fixed_attrs = []
            ran_t = random.randint(0, len(c_2opt)-1)
            # ran_a = random.randint(0, len(self.schema)-1)
            # fixed_attrs.append(self.schema[ran_a])
            t_content = dict(self.rep_csv.iloc[c_2opt[ran_t]])
            # tc_content = self.find_assignment(c_2opt[ran_t], fixed_attrs, c_2opt)
            while len(fixed_attrs) < len(self.schema):
                flag = 1
                while flag:
                    ran_a = random.randint(0, len(self.schema)-1)
                    if self.schema[ran_a] not in fixed_attrs:
                        flag = 0
                        fixed_attrs.append(self.schema[ran_a])
                tc_content = self.find_assignment(c_2opt[ran_t], fixed_attrs, c_2opt)
                if tc_content:
                    tc_content = dict(tc_content)
                    t_content.update(tc_content)
                else:
                    for attr, val in t_content.items():
                        self.rep_csv[attr][c_2opt[ran_t]] = val
                        if val != self.dirty_csv[attr][c_2opt[ran_t]]:
                            self.rep_cells.append((c_2opt[ran_t], self.schema.index(attr)))
            for attr, val in t_content.items():
                self.rep_csv[attr][c_2opt[ran_t]] = val
                if val != self.dirty_csv[attr][c_2opt[ran_t]]:
                        self.rep_cells.append((c_2opt[ran_t], self.schema.index(attr)))
            c_2opt.remove(c_2opt[ran_t])
    
    def data_diff(self, graph):
        diff_dict = {}
        for edg_idx, edge in graph.edgeList.items():
            head = edge.verList[0]
            tail = edge.verList[1]
            fd = edge.fd
            diff_list = []
            for attr in self.schema:
                if self.rep_csv[attr][head] != self.rep_csv[attr][tail]:
                    diff_list.append(attr)
            diff_list_str = "\t".join(diff_list)
            if diff_list_str not in diff_dict.keys():
                diff_dict[diff_list_str] = []
                diff_dict[diff_list_str].append((head, tail, fd))
            else:
                diff_dict[diff_list_str].append((head, tail, fd))
        return diff_dict
    
    def _compute_state_cost(self, state):
        cost = 0
        for change in state:
            cost += len(change)
        return cost

    def get_des_goal_states(self, si, sc, g, dc):
        if sc in visited_nodes.keys():
            return visited_nodes[sc]
        dc_backup = copy.deepcopy(dc)
        if len(dc) == 0:
            return [sc]
        des_states = []
        ran_d = list(dc.keys())[random.randint(0, len(dc)-1)]
        g_backup = copy.deepcopy(g)
        for diff in dc[ran_d]:
            g_backup.addEdge(diff[0], diff[1], diff[2])
        c_2opt = self.mvc(g_backup)
        if len(c_2opt)*min(len(self.schema)-1, len(self.ori_fd)) <= self.tau:
            dc_backup.pop(ran_d)
            des_states.extend(self.get_des_goal_states(si, sc, g_backup, dc_backup))
        child_state = self.tree.node_dict[sc].child
        for sta in child_state:
            # get the fdset corresponding to the sta
            sta_state = self.tree.node_dict[sta].state
            fd_set_i = copy.deepcopy(self.ori_fd)
            for s in range(len(sta_state)):
                if len(sta_state[s]) == 0:
                    continue
                else:
                    fd_set_i[s][0].extend(sta_state[s])
            # whether violation exists
            flag = 0
            for h, t, fd in dc[ran_d]:
                left = [i for i in fd_set_i[fd][0]]
                right = [i for i in fd_set_i[fd][1]]
                if all(self.rep_csv[left].values[h] == self.rep_csv[left].values[t]) and \
                    any(self.rep_csv[right].values[h] != self.rep_csv[right].values[t]):
                    flag = 1
                    break
            if flag:
                # if exists, continue
                continue
            # computing cells still violating the corresponding fdset
            for key in dc_backup.keys():
                for h, t, fd in dc_backup[key]:
                    # h, t, fd = dc_backup[key][j][0], dc_backup[key][j][1], dc_backup[key][j][2]
                    left = [i for i in fd_set_i[fd][0]]
                    right = [i for i in fd_set_i[fd][1]]
                    if (all(self.rep_csv[left].values[h] == self.rep_csv[left].values[t]) and \
                        all(self.rep_csv[right].values[h] == self.rep_csv[right].values[t])) or \
                            (any(self.rep_csv[left].values[h] != self.rep_csv[left].values[t]) and \
                                any(self.rep_csv[right].values[h] != self.rep_csv[right].values[t])):
                        dc_backup[key].remove((h, t, fd))
            graph = Graph()
            des_states.extend(self.get_des_goal_states(si, sta, graph, dc_backup))
        # remove non-minimal states
        res_dict = {}
        for state in des_states:
            sta_state = self.tree.node_dict[state].state
            fd_set_i = copy.deepcopy(self.ori_fd)
            for s in range(len(sta_state)):
                if len(sta_state[s]) == 0:
                    continue
                else:
                    fd_set_i[s][0].extend(sta_state[s])
            graph = self._build_graph(fd_set_i)
            mvc_h = self.mvc(graph)
            if len(mvc_h)*min(len(self.schema)-1, len(self.ori_fd)) <= self.tau:
                res_dict[state] = self._compute_state_cost(self.tree.node_dict[sc].state)
            # child_exist = [1 for ch in self.tree.node_dict[sc].child if ch in res_dict.keys()]
            # if len(child_exist) > 0:
            #     res_dict.pop(state)
        poped_id = []
        for state in res_dict.keys():
            child_exist = [1 for ch in self.tree.node_dict[sc].child if ch in res_dict.keys()]
            if len(child_exist) > 0:
                poped_id.append(state)
        for sta in poped_id:
            res_dict.pop(sta)
        des_states = list(res_dict.keys())
        # del dc_backup
        # gc.collect()
        visited_nodes[sc] = copy.deepcopy(des_states)
        return des_states
    
    def modify_fds(self, fd_set):
        logging.debug("Trying Building State Tree")
        self._build_state_tree()
        logging.debug("Finished Building State Tree")
        state_idx = 0
        pq = queue.PriorityQueue()
        pq.put((self._compute_state_cost(self.tree.node_dict[0].state), state_idx))
        while pq:
            state_idx = pq.get()[1]
            state = self.tree.node_dict[state_idx].state
            fd_set_h = copy.deepcopy(fd_set)
            for i in range(len(state)):
                if len(state[i]) == 0:
                    continue
                else:
                    fd_set_h[i][0].extend(state[i])
            graph = self._build_graph(fd_set_h)
            mvc_h = self.mvc(graph)
            diff_dict = self.data_diff(graph)
            if len(mvc_h)*min(len(fd_set), len(self.schema)) <= self.tau:
                return fd_set_h
            state_i = copy.deepcopy(self.tree.node_dict[state_idx].child)  # child state of state
            for si in state_i:
                min_cost = 1e10
                fd_set_i = copy.deepcopy(fd_set)
                si_state = self.tree.node_dict[si].state
                for i in range(len(si_state)):
                    if len(si_state[i]) == 0:
                        continue
                    else:
                        fd_set_i[0].extend(si_state[i])
                g = Graph()
                # lp = LineProfiler()
                # lp_wrapper = lp(self.get_des_goal_states)
                # min_states = lp_wrapper(si, si, g, diff_dict)
                min_states = self.get_des_goal_states(si, si, g, diff_dict)
                # lp.print_stats()
                if len(min_states) == 0:
                    pq.put((1e10, si))
                else:
                    for m_sta in min_states:
                        if self._compute_state_cost(self.tree.node_dict[m_sta].state) < min_cost:
                            min_cost = self._compute_state_cost(self.tree.node_dict[m_sta].state)
                    pq.put((min_cost, si))
        return None
                
                
    def find_assignment(self, t, fixed_attrs, c_2opt):
        out_opt_ver = [i for i in range(len(self.dirty_csv)) if i not in c_2opt]
        # attr_in_fd_idx = []
        # for idx, fd in self.cur_fd.items():
        #     for attr in fixed_attrs:
        #         if attr in fd[1] and len(fd[1]) == 1:
        #             attr_in_fd_idx.append(idx)
        tc_content = dict(self.rep_csv.iloc[t])
        flag_run = 1
        while flag_run:
            flag_run = 0
            for t_out in out_opt_ver:
                for fd_idx in range(len(self.cur_fd)):
                    left = self.cur_fd[fd_idx][0]
                    right = self.cur_fd[fd_idx][1]
                    if len(right) > 1:
                        continue
                    flag = 1
                    for attr in left:
                        if self.rep_csv[attr][t_out] != tc_content[attr]:
                            flag = 0
                            break
                    if flag:
                        if self.rep_csv[right].values[t_out] != tc_content[right[0]]:
                            if right[0] in fixed_attrs:
                                return None
                            else:
                                tout_right = dict(self.rep_csv.loc[t_out, right])
                                tc_content.update(tout_right)
                                for attr in right:
                                    fixed_attrs.append(attr)
                                fixed_attrs = list(set(fixed_attrs))
                                break
                if flag_run:
                    break  
                    
        return tc_content

    def run(self):
        starttime = datetime.datetime.now()
        logging.info("Prepare to Modify FDs ")
        # lp = LineProfiler()
        # lp_wrapper = lp(self.modify_fds)
        # self.cur_fd = lp_wrapper(self.ori_fd)
        # lp.print_stats()
        self.cur_fd = self.modify_fds(self.ori_fd)
        logging.info("Prepare to Repair Data ")
        self.repair_data()
        logging.info("Finish Repairing Data ")
        self.evaluation()
        # endtime = datetime.datetime.now()
        # print(endtime - starttime)
    
    def _build_state_tree(self):
        self.tree = Tree([[] for i in range(len(self.ori_fd))])
        self._build_child(0, self.tree)
        return self.tree
    
    def _build_child(self, parent, tree):
        par_state = tree.node_dict[parent].state
        cur_attrs = list(set([attri for si in par_state for attri in si]))
        max_attr_idx = 0
        if cur_attrs:
            max_attr_idx = max([self.schema.index(i) for i in cur_attrs])
            # if max_attr_idx == len(self.schema)-1:
            #     return None
        for i in range(len(par_state)):
            for attri in range(max_attr_idx, len(self.schema)):
                fd_attrs = copy.deepcopy(self.cur_fd[i][0])
                fd_attrs.extend(self.cur_fd[i][1])
                attr_set = []
                for k in range(0, i+1):
                    attr_set.extend(par_state[k])
                if self.schema[attri] in fd_attrs or self.schema[attri] in attr_set:
                    continue
                else:
                    child_state = copy.deepcopy(par_state)
                    child_state[i].append(self.schema[attri])
                    new_key = tree.add_node(parent, child_state)
                    self._build_child(new_key, tree)
                    # print_tree(tree)
        del par_state, cur_attrs
        # gc.collect()
    
    def _build_graph(self, fd_set):
        graph = Graph()
        for i in range(len(self.rep_csv)):
            for j in range(i+1, len(self.rep_csv)):
                 for fd_idx in range(len(fd_set)):
                    left = [i for i in fd_set[fd_idx][0]]
                    right = [i for i in fd_set[fd_idx][1]]
                    flag = 1
                    for attr in left:
                        if self.rep_csv[attr][i] != self.rep_csv[attr][j]:
                            flag = 0
                            break
                    if flag:
                        if any(self.rep_csv[right].values[i] != self.rep_csv[right].values[j]):
                            graph.addEdge(i, j, fd_idx)
        return graph

    def mvc(self, graph):
        ver_edg_cnt = {}
        edg_list = graph.getEdges()
        for ver in graph.vertList.keys():
            if ver not in ver_edg_cnt.keys():
                ver_edg_cnt[ver] = len(graph.vertList[ver].getConnections())
        ver_edg_cnt = sorted(ver_edg_cnt.items(), key=lambda kv: kv[1], reverse=True)
        ver_list = [i[0] for i in ver_edg_cnt]
        mvc_list = []
        cnt = 0
        while edg_list:
            for edg in graph.vertList[ver_list[cnt]].edge_list:
                if edg in edg_list:
                    edg_list.remove(edg)
            mvc_list.append(ver_list[cnt])
            cnt = cnt + 1
        return mvc_list
    
    def evaluation(self):
        self.rep_cells = list(set(self.rep_cells))
        self.wrong_cells = list(set(self.wrong_cells))
        if True:
            if not PERFECTED:
                det_right = 0
                out_path = "./Exp_result/Relative_trust/" + task_name[:-1] +"/onlyED_" + task_name + dirty_path[-25:-4] + ".txt"
                f = open(out_path, 'w')
                sys.stdout = f
                end_time = time.time()
                for cell in self.rep_cells:
                    if cell in self.wrong_cells:
                        det_right = det_right + 1
                pre = det_right / (len(self.rep_cells)+1e-10)
                rec = det_right / (len(self.wrong_cells)+1e-10)
                f1 = 2*pre*rec/(pre+rec+1e-10)
                print("{pre}\n{rec}\n{f1}\n{time}".format(pre=pre, rec=rec, f1=f1, time=(end_time-start_time)))
                f.close()

                out_path = "./Exp_result/Relative_trust/" + task_name[:-1] +"/oriED+EC_" + task_name + dirty_path[-25:-4] + ".txt"
                res_path = "./Repaired_res/Relative_trust/" + task_name[:-1] + "/repaired_" + task_name + dirty_path[-25:-4] + ".csv"
                self.rep_csv.to_csv(res_path, index=False, columns=list(self.rep_csv.columns))
                f = open(out_path, 'w')
                sys.stdout = f
                end_time = time.time()
                rep_right = 0
                rep_total = len(self.rep_cells)
                wrong_cells = len(self.wrong_cells)
                rec_right = 0
                for cell in self.rep_cells:
                    if self.dirty_csv.iloc[cell[0], cell[1]] == self.clean_csv.iloc[cell[0], cell[1]]:
                        rep_right += 1
                for cell in self.wrong_cells:
                    if self.dirty_csv.iloc[cell[0], cell[1]] == self.clean_csv.iloc[cell[0], cell[1]]:
                        rec_right += 1
                pre = rep_right / (rep_total+1e-10)
                rec = rec_right / (wrong_cells+1e-10)
                f1 = 2*pre*rec / (rec+pre+1e-10)
                print("{pre}\n{rec}\n{f1}\n{time}".format(pre=pre, rec=rec, f1=f1, time=(end_time-start_time)))
                f.close()
            else:
                out_path = "./Exp_result/Relative_trust/" + task_name[:-1] +"/prefectED+EC_" + task_name + dirty_path[-25:-4] + ".txt"
                res_path = "./Repaired_res/Relative_trust/" + task_name[:-1] + "/perfect_repaired_" + task_name + dirty_path[-25:-4] + ".csv"
                self.rep_csv.to_csv(res_path, index=False, columns=list(self.rep_csv.columns))
                f = open(out_path, 'w')
                sys.stdout = f
                end_time = time.time()
                rep_right = 0
                rep_total = len(self.rep_cells)
                wrong_cells = len(self.wrong_cells)
                rec_right = 0
                rep_t = 0
                for cell in self.wrong_cells:
                    if cell in self.rep_cells:
                        rep_t += 1
                        if self.dirty_csv.iloc[cell[0], cell[1]] == self.clean_csv.iloc[cell[0], cell[1]]:
                            rec_right += 1
                pre = rec_right / (rep_t+1e-10)
                rec = rec_right / (wrong_cells+1e-10)
                f1 = 2*pre*rec / (rec+pre+1e-10)
                print("{pre}\n{rec}\n{f1}\n{time}".format(pre=pre, rec=rec, f1=f1, time=(end_time-start_time)))
                f.close()


if __name__ == "__main__":
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
    
    # dirty_path = "/data/nw/DC_ED/References_inner_and_outer/SCAREd/dirty.csv"
    # dirty_path = "./data with dc_rules/flights/noise/flights-outer_error-10.csv"
    # rule_path = "./data with dc_rules/flights/dc_rules-validate-fd-horizon.txt"
    # clean_path = "./data with dc_rules/flights/clean.csv"
    # clean_path = "/data/nw/DC_ED/References_inner_and_outer/SCAREd/clean.csv"
    # task_name = "flights1"
    # ONLYED = 1
    # PERFECTED = 0

    start_time = time.time()
    logging.basicConfig(level=logging.DEBUG)
    R_Clean = RelaTrust(dirty_path, clean_path, rule_path)
    R_Clean.run()