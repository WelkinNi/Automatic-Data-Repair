from random import choice, randint

from networkx import *
from networkx.algorithms.isolate import isolates


class MinimumVertexCoverSolver:

    @staticmethod
    def is_circle(graph):
        for _, degree in graph.degree:
            if degree != 2:
                return False
        return True

    @staticmethod
    def is_path(graph: Graph):
        number_of_1_degrees = 0
        for _, degree in graph.degree:
            if not (1 <= degree <= 2):
                return False
            if degree == 1:
                number_of_1_degrees += 1
        return number_of_1_degrees == 2

    @staticmethod
    def is_complete(graph: Graph):
        n = graph.number_of_nodes()
        e = graph.number_of_edges()
        return n * (n - 1) / 2 == e

    @staticmethod
    def min_vertex_cover_path(graph: Graph):
        min_vertex = set()
        start_node, end_node = (n for n, d in graph.degree if d == 1)
        current_node = start_node
        flag = False
        visited = set()
        while True:
            if flag:
                min_vertex.add(current_node)
            if current_node == end_node:
                break
            visited.add(current_node)
            for node in graph.neighbors(current_node):
                if node not in visited:
                    current_node = node

            flag = not flag

        return min_vertex

    @staticmethod
    def min_vertex_cover_circle(graph: Graph):
        flag = False
        min_vertex = set(n for n in graph.nodes if (flag := not flag))
        return min_vertex

    @staticmethod
    def min_vertex_cover_complete(graph: Graph):
        remove_index = randint(0, graph.number_of_nodes() - 1)
        return set(n for i, n in enumerate(graph.nodes) if i != remove_index)

    def __call__(self, g, init_node=None):
        if not init_node:
            # current_node = sorted(g.degree, key=lambda x: x[1], reverse=True)[0][0]
            current_node = choice(list(g.nodes))
        else:
            current_node = init_node

        s = {current_node}

        while True:
            h_graph = self.remove_nodes_from_graph(g, s)
            if number_of_isolates(h_graph) == h_graph.number_of_nodes():
                return s

            is_continue = False
            desired_connected_components = [cc for cc in connected_components(h_graph) if len(cc) > 1]
            for connected_component_set in desired_connected_components:
                cc_graph = subgraph(h_graph, connected_component_set)
                if self.is_complete(cc_graph):
                    is_continue = True
                    cc_s = self.min_vertex_cover_complete(cc_graph)
                elif self.is_circle(cc_graph):
                    is_continue = True
                    cc_s = self.min_vertex_cover_circle(cc_graph)
                elif self.is_path(cc_graph):
                    is_continue = True
                    cc_s = self.min_vertex_cover_path(cc_graph)
                else:
                    cc_s = {}
                # cc_s = self(cc_graph)
                s = s.union(cc_s)

            if is_continue:
                continue

            nodes_with_degree_one = [node for node in h_graph.nodes if h_graph.degree[node] == 1]
            if nodes_with_degree_one:
                for node in nodes_with_degree_one:
                    adj_node = next(h_graph.neighbors(node))
                    # current_node = adj_node
                    s.add(adj_node)
                continue

            h_graph = self.remove_nodes_from_graph(g, s - {current_node})
            allowed_vertices = set(g.nodes) - s.union(isolates(h_graph))

            if current_node in isolates(h_graph):
                current_node = choice(list(allowed_vertices))
                s.add(current_node)
                continue

            shortest_distances = {}
            for destination_node in allowed_vertices:
                try:
                    shortest_distances[destination_node] = dijkstra_path_length(h_graph, current_node, destination_node)
                except NetworkXNoPath:
                    pass

            d = max(shortest_distances.values())

            if d > 1:
                t = [item[0] for item in shortest_distances.items() if item[1] == d - 1]

                if len(t) == 1:
                    current_node = t.pop()
                    s.add(current_node)
                    continue

                node_cc_increase = {}
                for node in t:
                    g_temp = self.remove_nodes_from_graph(h_graph, isolates(h_graph))
                    before_cc_count = number_connected_components(g_temp)

                    g_temp = self.remove_nodes_from_graph(h_graph, {node})
                    g_temp = self.remove_nodes_from_graph(g_temp, isolates(g_temp))
                    after_cc_count = number_connected_components(g_temp)
                    node_cc_increase[node] = after_cc_count - before_cc_count

                min_value = min(node_cc_increase.values())
                nodes_with_min_cc_increase = [node[0] for node in node_cc_increase.items() if node[1] == min_value]
                if len(nodes_with_min_cc_increase) == 1:
                    current_node = nodes_with_min_cc_increase.pop()

                else:
                    nodes_degree = {node: h_graph.degree[node] for node in nodes_with_min_cc_increase}
                    max_degree = max(nodes_degree.values())
                    max_degree_nodes_list = [item[0] for item in nodes_degree.items() if item[1] == max_degree]
                    current_node = min(node for node in max_degree_nodes_list)

                s.add(current_node)
                continue

            else:
                h_graph = self.remove_nodes_from_graph(g, s)
                is_min_cover = True
                for cc_graph in connected_components(h_graph):
                    cc_graph = h_graph.subgraph(cc_graph)
                    if self.is_complete(cc_graph) or self.is_path(cc_graph) or self.is_circle(cc_graph):
                        continue
                    else:
                        is_min_cover = False
                        break

                if is_min_cover:
                    return s

                nodes_degree = dict(h_graph.degree)
                max_degree = max(nodes_degree.values())
                max_degree_nodes_list = [item[0] for item in nodes_degree.items() if item[1] == max_degree]

                current_node = min(node for node in max_degree_nodes_list)
                s.add(current_node)
                continue

    @staticmethod
    def remove_nodes_from_graph(g, n):
        """
        create a deepcopy from graph `g` and remove given nodes.
        """
        must_be_nodes = set(g.nodes) - set(n)
        new_g: Graph = subgraph(g, must_be_nodes)
        return new_g