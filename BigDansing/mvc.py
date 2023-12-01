from tes import greedy_min_vertex_cover
import re
def read_graph_dc(graph):
    v = {}
    for edge in range(len(graph)):
        for i in range(len(graph[edge])):
            #0中存放的指示是那条规则导致的违规
            if(i==0):continue
            #1,4,7....中存放为后两个因为哪个操作符违规
            if(i%3==1):continue
            try:
                v[graph[edge][i]].append(edge)
            except:
                v[graph[edge][i]] = [edge]
    return v

def read_graph(graph):
    '''Reads a text file containing a bipartite graph
    where the vertices are represented by integers. The
    output contains two dictionaries formatted
    appropriately for use with the min_vertex_cover
    function. The text file must contain two columns
    of integers, with each column separted by a space.
    Each column of integers must contain a range of
    integers that does not overlap with the adjacent column.

    For example:

    1000 2000
    1001 2000
    1002 2001
    1003 2002
    1004 2000
    1004 2001
    1005 2003
    1006 2004

    Each row is equivalent to an edge. So, 2000 has three
    edges leading to 1000, 1001, and 1004 and 1000 has
    only one edge, leadng to 2000.'''

    left_v = {}
    right_v = {}
    for edge in graph:
        try:
            left_v[edge[0]].append(edge[1])
        except:
            left_v[edge[0]] = [edge[1]]
        try:
            right_v[edge[1]].append(edge[0])
        except:
            right_v[edge[1]] = [edge[0]]
    return [left_v, right_v]


# Hopcroft-Karp bipartite max-cardinality matching and max independent set
# David Eppstein, UC Irvine, 27 Apr 2002

def bipartiteMatch(graph):
    '''Find maximum cardinality matching of a bipartite graph (U,V,E).
    The input format is a dictionary mapping members of U to a list
    of their neighbors in V.  The output is a triple (M,A,B) where M is a
    dictionary mapping members of V to their matches in U, A is the part
    of the maximum independent set in U, and B is the part of the MIS in V.
    The same object may occur in both U and V, and is treated as two
    distinct vertices if this happens.'''

    # initialize greedy matching (redundant, but faster than full search)
    matching = {}
    for u in graph:
        for v in graph[u]:
            if v not in matching:
                matching[v] = u
                break
    while 1:
        # structure residual graph into layers
        # pred[u] gives the neighbor in the previous layer for u in U
        # preds[v] gives a list of neighbors in the previous layer for v in V
        # unmatched gives a list of unmatched vertices in final layer of V,
        # and is also used as a flag value for pred[u] when u is in the first layer
        preds = {}
        unmatched = []
        pred = dict([(u, unmatched) for u in graph])
        for v in matching:
            del pred[matching[v]]
        layer = list(pred)

        # repeatedly extend layering structure by another pair of layers
        while layer and not unmatched:
            newLayer = {}
            for u in layer:
                for v in graph[u]:
                    if v not in preds:
                        newLayer.setdefault(v, []).append(u)
            layer = []
            for v in newLayer:
                preds[v] = newLayer[v]
                if v in matching:
                    layer.append(matching[v])
                    pred[matching[v]] = v
                else:
                    unmatched.append(v)

        # did we finish layering without finding any alternating paths?
        if not unmatched:
            unlayered = {}
            for u in graph:
                for v in graph[u]:
                    if v not in preds:
                        unlayered[v] = None
            return (matching, list(pred), list(unlayered))

        # recursively search backward through layers to find alternating paths
        # recursion returns true if found path, false otherwise
        def recurse(v):
            if v in preds:
                L = preds[v]
                del preds[v]
                for u in L:
                    if u in pred:
                        pu = pred[u]
                        del pred[u]
                        if pu is unmatched or recurse(pu):
                            matching[v] = u
                            return 1
            return 0

        for v in unmatched: recurse(v)


# Find a minimum vertex cover
def min_vertex_cover(left_v, right_v):
    '''Use the Hopcroft-Karp algorithm to find a maximum
    matching or maximum independent set of a bipartite graph.
    Next, find a minimum vertex cover by finding the
    complement of a maximum independent set.

    The function takes as input two dictionaries, one for the
    left vertices and one for the right vertices. Each key in
    the left dictionary is a left vertex with a value equal to
    a list of the right vertices that are connected to the key
    by an edge. The right dictionary is structured similarly.

    The output is a dictionary with keys equal to the vertices
    in a minimum vertex cover and values equal to lists of the
    vertices connected to the key by an edge.

    For example, using the following simple bipartite graph:

    1000 2000
    1001 2000

    where vertices 1000 and 1001 each have one edge and 2000 has
    two edges, the input would be:

    left = {1000: [2000], 1001: [2000]}
    right = {2000: [1000, 1001]}

    and the ouput or minimum vertex cover would be:

    {2000: [1000, 1001]}

    with vertex 2000 being the minimum vertex cover.

    The code can also generate a bipartite graph with an arbitrary
    number of edges and vertices, write the graph to a file, and
    read the graph and convert it to the appropriate format.'''
    data_hk = bipartiteMatch(left_v)
    left_mis = data_hk[1]
    right_mis = data_hk[2]
    mvc = left_v.copy()
    for i in list(right_v):
        if(mvc.__contains__(i)):
            mvc[i].extend(right_v[i])
            del(right_v[i])
    mvc.update(right_v)
    # print("---------------mvc-------------------")
    # print(mvc)
    mvc = greedy_min_vertex_cover(mvc)
    # print(mvc)
    return mvc
    # for v in left_mis:
    #     del (mvc[v])
    # for v in right_mis:
    #     try:
    #         del (mvc[v])
    #     except:
    #         print("can't delete",v)
    #         continue
    # print(mvc)
    # return mvc

# output_data = min_vertex_cover(input_data[0], input_data[1])

# print(output_data)
