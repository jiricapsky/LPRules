from networkx import MultiDiGraph


def bfs(start: str, target: str, graph: MultiDiGraph, max_length=5, skipped_r='', n=0):
    """
    Method for finding shortest path in graph using breadth first search. If specified n additional paths (each longer
    than the previous one) are returned. Length of paths is limited by maximal length with default value 5.
    :param start: name of starting node
    :param target: name of target node
    :param graph: graph where paths are searched
    :param max_length: maximal length of path
    :param skipped_r: relation that is skipped for first edge of path
    :param n: number of additional paths, each is than previous one
    :return: paths from 'start' to 'target' in MultiDiGraph 'graph' as list of edges
    """
    visited = []
    queue = [[start]]
    result = []
    longer_paths_remaining = n

    length = 1

    while (length <= max_length) & (len(queue) > 0):
        path = queue.pop(0)
        node = path[-1]
        # if node is edge from previous loop, select only second entity
        if node.__class__ == tuple:
            node = node[1]

        if node not in visited:
            # check all neighbors
            for neighbour in [n for n in graph.edges(node, data='relation')]:
                # skip neighbor with selected relation for length one path
                if (length == 1) & (neighbour[2] == skipped_r):
                    continue
                else:
                    # add to path
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)

                    # return neighbour if the same as target
                    # represented as edge in format: (X, Y, relation)
                    if neighbour[1] == target:
                        # skip if the path has the same length as the previous result
                        # lenght -1 is used because first element of path is the start node
                        if len(result) > 0:
                            if len(result[-1]) >= (len(new_path)-1):
                                continue
                            else:
                                longer_paths_remaining -= 1

                        result.append(new_path[1:])
                        # print(longer_paths_remaining, " ", start, "->", target, '(', skipped_r, ") : ", new_path[1:])

                        # end loop if no more paths need to be found
                        if longer_paths_remaining <= 0:
                            break

            visited.append(node)

        # check if additional longer paths have to be found and shortest path has been found
        if (len(result) > 0) & (longer_paths_remaining <= 0):
            return result

        length += 1

    return []
