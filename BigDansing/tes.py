import  heapq

def greedy_min_vertex_cover(graph: dict):

    # queue used to store nodes and their rank
    queue: list[list] = []

    # for each node and his adjacency list add them and the rank of the node to queue
    # using heapq module the queue will be filled like a Priority Queue
    # heapq works with a min priority queue, so I used -1*len(v) to build it
    for key, value in graph.items():
        # O(log(n))
        heapq.heappush(queue, [-1 * len(value), (key, value)])

    # print("----------------")
    # print(queue)
    # chosen_vertices = set of chosen vertices
    chosen_vertices = set()

    # while queue isn't empty and there are still edges
    #   (queue[0][0] is the rank of the node with max rank)
    while queue and queue[0][0] != 0:
        # extract vertex with max rank from queue and add it to chosen_vertices
        argmax = heapq.heappop(queue)[1][0]
        chosen_vertices.add(argmax)

        # Remove all arcs adjacent to argmax
        for elem in queue:
            # if v haven't adjacent node, skip
            if elem[0] == 0:
                continue
            # if argmax is reachable from elem
            # remove argmax from elem's adjacent list and update his rank
            if argmax in elem[1][1]:
                index = elem[1][1].index(argmax)
                del elem[1][1][index]
                elem[0] += 1
        # re-order the queue
        heapq.heapify(queue)
    return chosen_vertices

def greedy_min_vertex_cover_dc(graph: dict,vio):

    # queue used to store nodes and their rank
    queue: list[list] = []

    # for each node and his adjacency list add them and the rank of the node to queue
    # using heapq module the queue will be filled like a Priority Queue
    # heapq works with a min priority queue, so I used -1*len(v) to build it
    for key, value in graph.items():
        # O(log(n))
        heapq.heappush(queue, [-1 * len(value), (key, value)])

    # print("----------------")
    # print(queue)
    # chosen_vertices = set of chosen vertices
    chosen_vertices = set()

    # while queue isn't empty and there are still edges
    #   (queue[0][0] is the rank of the node with max rank)
    while queue and queue[0][0] != 0:
        # print(len(queue))
        # extract vertex with max rank from queue and add it to chosen_vertices
        argmax = heapq.heappop(queue)[1][0]
        chosen_vertices.add(argmax)

        # Remove all arcs adjacent to argmax
        for elem in queue:
            # if v haven't adjacent node, skip
            if elem[0] == 0:
                continue
            # if argmax is reachable from elem
            # remove argmax from elem's adjacent list and update his rank
            for i in elem[1][1]:
                if argmax in vio[i]:
                    index = elem[1][1].index(i)
                    del elem[1][1][index]
                    elem[0] += 1
        # re-order the queue
        heapq.heapify(queue)
    return chosen_vertices

if __name__ == "__main__":
    import doctest

    doctest.testmod()

    graph = {
            0: [1, 3], 1: [0, 3], 2: [0, 3, 4], 3: [0, 1, 2], 4: [2, 3]}
    print(greedy_min_vertex_cover(graph))