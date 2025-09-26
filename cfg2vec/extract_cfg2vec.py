import pandas as pd
import networkx as nx
import matplotlib.pylab as plt
import numpy as np
import os, json, math, csv
from parse.cfg2vec import Cfg2Vec
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler

# analysedcfglists cfglists
cfgpath = "../dataset/graphlists/filteredcfg_merge_graph.csv"
xiaorongcfgpath = "../dataset/graphlists/filteredcfg_nomerge_graph.csv"
blockpath = "../dataset/blocks/identifiedcfgs.csv"

featuespath = "../dataset/features/"
vecfeatuespath = "../dataset/features/cfg2vec/"
ablationpath = "../dataset/features/ablation/"


def extract_cfg2vec(featurefile, epoch):
    data = pd.read_csv(cfgpath, index_col="address")
    print("load data finished")

    success = 0
    total = len(data.index)
    fail = 0
    faillist = []

    veccol = ['vec_' + str(i) for i in range(100)]
    cols = ['address'] + veccol
    df = pd.DataFrame(columns=cols)

    graphs = []
    map = {}
    n = []

    for ind in data.index:
        try:
            nodelist = json.loads(data.loc[ind, "nodes"])
            edgelist = json.loads(data.loc[ind, "edges"])
            g = nx.DiGraph()
            g.add_nodes_from(nodelist)
            g.add_edges_from(edgelist)
            if len(g) == 0:
                n.append(ind)
            graphs.append(g)
            map[ind] = success
            success += 1

        except Exception as e:
            print(ind, e)
            faillist.append(ind)
            fail += 1
        finally:
            print("get graph: ", success, " / ", total, " / ", fail)

    print("get graphs total:", len(graphs))
    print(n)

    model = Cfg2Vec(dimensions=100, wl_iterations=epoch)
    model.fit(graphs)
    vecs = model.get_embedding()
    print("fit finished")

    success = 0
    total = len(data.index)
    fail = 0
    faillist = []

    for ind in data.index:
        if ind in map.keys():
            vi = vecs[map[ind]]
            v = vi.tolist()
            newline = dict(zip(veccol, v))
            newline['address'] = ind
            newrow = pd.Series(newline)
            df = df.append(newrow, ignore_index=True)
            success += 1
            print("get vec: ", success, " / ", total)
        else:
            print(ind)

    df.to_csv(vecfeatuespath + featurefile, index=False)
    print(faillist)


def extract_xiaorong_cfg2vec(featurefile, epoch):
    data = pd.read_csv(xiaorongcfgpath, index_col="address")
    print("load data finished")

    success = 0
    total = len(data.index)
    fail = 0
    faillist = []

    veccol = ['vec_' + str(i) for i in range(100)]
    cols = ['address'] + veccol
    df = pd.DataFrame(columns=cols)

    graphs = []
    map = {}
    n = []

    for ind in data.index:
        try:
            nodelist = json.loads(data.loc[ind, "nodes"])
            edgelist = json.loads(data.loc[ind, "edges"])
            g = nx.DiGraph()
            g.add_nodes_from(nodelist)
            g.add_edges_from(edgelist)
            if len(g) == 0:
                n.append(ind)
            graphs.append(g)
            map[ind] = success
            success += 1

        except Exception as e:
            print(ind, e)
            faillist.append(ind)
            fail += 1
        finally:
            print("get graph: ", success, " / ", total, " / ", fail)

    print("get graphs total:", len(graphs))
    print(n)

    model = Cfg2Vec(dimensions=100, wl_iterations=epoch, use_path=False)
    model.fit(graphs)
    vecs = model.get_embedding()
    print("fit finished")

    success = 0
    total = len(data.index)
    fail = 0
    faillist = []

    for ind in data.index:
        if ind in map.keys():
            vi = vecs[map[ind]]
            v = vi.tolist()
            newline = dict(zip(veccol, v))
            newline['address'] = ind
            newrow = pd.Series(newline)
            df = df.append(newrow, ignore_index=True)
            success += 1
            print("get vec: ", success, " / ", total)
        else:
            print(ind)

    df.to_csv(ablationpath + featurefile, index=False)
    print(faillist)


def _dfs(graph, node, visited, paths):
    print(node, graph.in_degree(node), [i for i in graph.successors(node)])
    visited.append(node)
    print(visited)
    if graph.out_degree(node) == 0:
        paths.append(visited.copy())
        print(paths)
    else:
        for i in graph.successors(node):
            if i not in visited:
                _dfs(graph, i, visited, paths)
            c = visited.count(i)
            if c < graph.in_degree(i):
                _dfs(graph, i, visited, paths)
    visited.pop()


if __name__ == "__main__":
    extract_cfg2vec("cfg2vec_1_1.csv", 1)
    extract_cfg2vec("cfg2vec_2_1.csv", 2)
    extract_cfg2vec("cfg2vec_3_1.csv", 3)
    extract_cfg2vec("cfg2vec_4_1.csv", 4)
    extract_cfg2vec("cfg2vec_5_1.csv", 5)
    extract_cfg2vec("cfg2vec_6_1.csv", 6)
    extract_cfg2vec("cfg2vec_7_1.csv", 7)
    extract_cfg2vec("cfg2vec_8_1.csv", 8)
    extract_cfg2vec("cfg2vec_9_1.csv", 9)
    extract_cfg2vec("cfg2vec_10_1.csv", 10)

    extract_xiaorong_cfg2vec("xiaorong_cfg2vec_nomerge_4.csv", 4)
    extract_xiaorong_cfg2vec("xiaorong_cfg2vec_nomerge_5.csv", 5)

    extract_xiaorong_cfg2vec("xiaorong_cfg2vec_nopath_4.csv", 4)
    extract_xiaorong_cfg2vec("xiaorong_cfg2vec_nopath_5.csv", 5)
