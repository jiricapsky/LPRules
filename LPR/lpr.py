import os.path
import re

import networkx
import matplotlib.pyplot as plt
from networkx import MultiDiGraph
from LPR.data.model import Model
from LPR.data.rules import Rules
from LPR.utils import split_dataset

tail = 0
head = 1


def edges_r(relation, graph: MultiDiGraph):
    edges = set()
    for u, v, r in graph.edges(data="relation"):
        if r == relation:
            edges.add((u, v))
    return edges


def get_relations(graph: MultiDiGraph, onlyOriginal=False):
    relations = set()
    for edge in graph.edges(data='relation'):
        if onlyOriginal & (not is_original_relation(edge[2])):
            continue
        relations.add(edge[2])
    return relations


def is_original_relation(relation: str):
    if bool(re.search(r'\bR_.*', relation)):
        return False
    return True


def create_reverse_relations(graph: MultiDiGraph):
    reversed = MultiDiGraph()
    for u, v, relation in graph.edges(data="relation"):
        reversed.add_edge(v, u, relation="R_{0}".format(relation))
    return reversed


def graphFromArr(data: []):
    """
    create MultiDiGraph from data
    :param data: input data where each edge has form [tail, relation, head]
    :return: graph and set of unique relations
    """
    graph = MultiDiGraph()
    relations = set()
    for d in data:
        graph.add_edge(d[0], d[2], relation=d[1])
        relations.add(d[1])

    return graph, relations


def getEntityRank(items: list[tuple[str, int]], target):
    for item in items:
        if item[0] == target:
            return items.index(item) + 1
    return None


def showGraph(graph: MultiDiGraph):
    relations = {(edge[0], edge[1]): edge[2] for edge in graph.edges(data='relation')}
    pos = networkx.spring_layout(graph)

    networkx.draw(graph, pos, with_labels=True)
    networkx.draw_networkx_edge_labels(graph, pos, edge_labels=relations, verticalalignment="baseline")
    plt.show()


def getMRR(ranks: [float]):
    mrr = 0
    for rank in ranks:
        mrr += 1 / rank

    mrr = mrr / len(ranks)
    return mrr


def getHITS_k(ranks: [], k: int):
    hits = 0
    for rank in ranks:
        if rank <= k:
            hits += 1

    hits = hits / len(ranks)
    return hits


def predict(data: MultiDiGraph, all_rules: [Rules], options=None):
    # prepare variables from test dataset
    graph = data
    relations = get_relations(graph, onlyOriginal=True)

    # create prediction function for original relations in training set
    functions = dict()
    for rules in all_rules:
        functions[rules.relation] = rules

    # create query (t,r,?) and (?,r,h) for each fact
    ranks = []
    for fact in [edges for edges in graph.edges(data='relation') if is_original_relation(edges[2])]:
        try:
            ranks.append(triple_rank(fact, functions[fact[2]], graph))
        except KeyError:
            # print("No rules for relation '{0}' found".format(fact[2]))
            pass

    if len(ranks) == 0:
        return 0

    if options is not None:
        if 'only_mrr' in options:
            return getMRR(ranks)

    mrr = getMRR(ranks)
    hits_1 = getHITS_k(ranks, 1)
    hits_3 = getHITS_k(ranks, 3)
    hits_10 = getHITS_k(ranks, 10)

    # print("\tMRR\n{0}\n\tHITS\n1  : {1}\n3  : {2}\n10 : {2}".format(mrr, hits_1, hits_3, hits_10))
    return {'mrr': mrr, 'hits_1': hits_1, 'hits_3': hits_3, 'hits_10': hits_10}
    # showGraph(test)


def triple_rank(fact: tuple[str, str, str], rules: Rules, graph: MultiDiGraph):
    right_predict = []
    left_predict = []

    for node in graph.nodes:
        # calculate score for each triple (t,r,e) and (e,r,h) where 'e' is the evaluated entity
        if node != fact[0]:
            right_predict.append(
                (node, entity_score((fact[0], node, fact[2]), rules, graph, eval_entity=head)))
        if node != fact[1]:
            left_predict.append((node, entity_score((node, fact[1], fact[2]), rules, graph, eval_entity=tail)))

    right_predict.sort(reverse=True, key=lambda entity_score: entity_score[1])
    left_predict.sort(reverse=True, key=lambda entity_score: entity_score[1])

    true_right_rank = getEntityRank(right_predict, fact[1])
    true_left_rank = getEntityRank(left_predict, fact[0])
    result = (true_right_rank + true_left_rank) / 2
    # print("---")
    # print(fact[1], true_right_rank, " : ", right_predict)
    # print(fact[0], true_left_rank, " : ", left_predict)

    return result


def entity_score(triple: tuple[str, str, str], rules: Rules, graph: MultiDiGraph, eval_entity=head):
    # calculate score for each rule
    score = 0
    reverse = False
    if eval_entity == tail:
        reverse = True

    for id in rules.rules[triple[2]]:
        if rules.is_unknown_path(id, (triple[0], triple[1]), graph, reverse):
            score += rules.rules[triple[2]][id]['weight']

    return score


class LPR:
    """single file - split into 'Test', 'Validate' and 'Train'"""

    def __init__(self, filename, train_size=0.6, test_size=0.2):
        if os.path.isfile(filename):
            train, test, validate = split_dataset.split(filename, train_size, test_size)
            self.train, self.train_r = graphFromArr(train)
            self.test, self.test_r = graphFromArr(test)
            self.validate, self.validate_r = graphFromArr(validate)

            # add reverse relations
            reversed = create_reverse_relations(self.train)
            self.train.add_edges_from(reversed.edges(data=True))


            # add reverse relations
            reversed = create_reverse_relations(self.test)
            self.test.add_edges_from(reversed.edges(data=True))

            # add reverse relations
            reversed = create_reverse_relations(self.validate)
            self.validate.add_edges_from(reversed.edges(data=True))

    def fit(self, trainData: MultiDiGraph, validationData: MultiDiGraph):
        train = trainData
        train_r = get_relations(train, onlyOriginal=True)
        validation = validationData
        validation_r = get_relations(validation, onlyOriginal=True)

        len0 = len(list(train.edges(data=True)))

        """add reverse relations"""
        # reversed = create_reverse_relations(train)
        #
        # train.add_edges_from(reversed.edges(data=True))

        models = []
        for original_relation in train_r:
            edges = edges_r(original_relation, train)
            models.append(Model(original_relation, train, edges))

        # train each model
        for model in models:
            # full t for kinship =  0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06
            # number of iterations should be 20
            model.train(validation, iterations=20, all_t=[0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06])

        return [model.rules for model in models]

    # def predict(self, data: MultiDiGraph, all_rules: [Rules]):
    #     # prepare variables from test dataset
    #     graph = data
    #     relations = get_relations(graph, onlyOriginal=True)
    #
    #     # add reverse relations
    #     reversed = create_reverse_relations(graph)
    #     graph.add_edges_from(reversed.edges(data=True))
    #
    #     # create prediction function for original relations in training set
    #     functions = dict()
    #     for rules in all_rules:
    #         functions[rules.relation] = rules
    #
    #     # create query (t,r,?) and (?,r,h) for each fact
    #     ranks = []
    #     for fact in [edges for edges in graph.edges(data='relation') if is_original_relation(edges[2])]:
    #         try:
    #             ranks.append(self.triple_rank(fact, functions[fact[2]], graph))
    #         except KeyError:
    #             print("No rules for relation '{0}' found".format(fact[2]))
    #
    #     mrr = getMRR(ranks)
    #     hits_1 = getHITS_k(ranks, 1)
    #     hits_3 = getHITS_k(ranks, 3)
    #     hits_10 = getHITS_k(ranks, 10)
    #
    #     print("\tMRR\n{0}\n\tHITS\n1  : {1}\n3  : {2}\n10 : {2}".format(mrr, hits_1, hits_3, hits_10))
    #     return mrr
    #     # showGraph(test)
    #
    # def triple_rank(self, fact: tuple[str, str, str], rules: Rules, graph: MultiDiGraph):
    #     right_predict = []
    #     left_predict = []
    #
    #     for node in graph.nodes:
    #         # calculate score for each triple (t,r,e) and (e,r,h) where 'e' is the evaluated entity
    #         if node != fact[0]:
    #             right_predict.append(
    #                 (node, self.entity_score((fact[0], node, fact[2]), rules, graph, eval_entity=head)))
    #         if node != fact[1]:
    #             left_predict.append((node, self.entity_score((node, fact[1], fact[2]), rules, graph, eval_entity=tail)))
    #
    #     right_predict.sort(reverse=True, key=lambda entity_score: entity_score[1])
    #     left_predict.sort(reverse=True, key=lambda entity_score: entity_score[1])
    #
    #     true_right_rank = getEntityRank(right_predict, fact[1])
    #     true_left_rank = getEntityRank(left_predict, fact[0])
    #     triple_rank = (true_right_rank + true_left_rank) / 2
    #     # print("---")
    #     # print(fact[1], true_right_rank, " : ", right_predict)
    #     # print(fact[0], true_left_rank, " : ", left_predict)
    #
    #     return triple_rank
    #
    # def entity_score(self, triple: tuple[str, str, str], rules: Rules, graph: MultiDiGraph, eval_entity=head):
    #     # calculate score for each rule
    #     score = 0
    #     reverse = False
    #     if eval_entity == tail:
    #         reverse = True
    #
    #     for id in rules.rules[triple[2]]:
    #         if rules.is_unknown_path(id, (triple[0], triple[1]), graph, reverse):
    #             score += rules.rules[triple[2]][id]['weight']
    #
    #     return score
