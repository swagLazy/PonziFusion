import networkx as nx
import pandas as pd
import os,json,math,csv

THRESHOLD = 50

blockpath = "../../dataset/blocks/filteredcfgs.csv"
listspath = "../../dataset/graphlists/"


def generatesimpleGraph(basicblocks):
    g = nx.DiGraph()
    maps = {}
    i = 0

    for block in basicblocks:
        maps[block["offset"]] = i
        g.add_node(i)
        i += 1
    for block in basicblocks:
        for suc in block["successors"]:
            print(type(block["offset"]))
            g.add_edge(maps[block["offset"]], maps[suc])
    return g



def generatefilteredGraph(basicblocks):
    g = nx.DiGraph()
    maps = {}
    i = 0
    removednode = []

    for block in basicblocks:
        maps[block["offset"]] = i
        g.add_node(i)
        i += 1
        if block["retain"] == False:
            removednode.append(maps[block["offset"]])
    for block in basicblocks:
        for suc in block["successors"]:
            print(type(block["offset"]))
            g.add_edge(maps[block["offset"]], maps[suc])
    for node in removednode:
        removenode(g, node)
    return g



def removenode(h, node):
    if node in h:
        innode = list(h.predecessors(node))
        outnode = list(h.successors(node))
        for i in innode:
            for j in outnode:
                if i != j:
                    h.add_edge(i, j)
        h.remove_node(node)



def deep(G, H, pre, now, visited):
    if now in visited:
        H.add_edge(pre, now)
        return
    else:
        visited.append(now)
    if G.in_degree(now) == 1 and G.out_degree(now) == 1 and G.in_degree(list(G.successors(now))[0]) == 1:
        deep(G, H, pre, list(G.successors(now))[0], visited)
    else:
        if pre is None:
            H.add_node(now)
        else:
            H.add_edge(pre, now)
        if G.out_degree(now) != 0:
            for i in G.successors(now):
                deep(G, H, now, i, visited)



def relabelGraph(G):
    h = nx.DiGraph()
    maps = dict()
    i = 0

    for node in G.nodes():
        maps[node] = i
        h.add_node(i)
        i += 1
    for node in G.nodes():
        for suc in G.successors(node):
            h.add_edge(maps[node], maps[suc])
    return h



def merge_nodes(G):
    H = nx.DiGraph()
    node = list(G.nodes)[0]
    deep(G, H, None, node, [])
    return H


def getGraph(basicblocks):
    
    g = generatefilteredGraph(basicblocks)
    if len(g) > THRESHOLD:
        g = merge_nodes(g)
    r = relabelGraph(g)
    return r


def extract_cfglist(file):
    data = pd.read_csv(blockpath, index_col="address")
    print("load data finished")

    success = 0
    total = len(data.index)
    fail = 0
    faillist = []
    sh = []

    
    rows_list = []

    for ind in data.index:
        try:
            blockjson = json.loads(data.loc[ind, "basicblocks"])
            g = getGraph(blockjson["basicblocks"])
            if len(g) == 0:
                sh.append(ind)

            
            newline = {
                "address": ind,
                "nodes": json.dumps(list(g.nodes)),
                "edges": json.dumps(list(g.edges))
            }

            
            rows_list.append(newline)

            success += 1

        except Exception as e:
            print(f"Error processing {ind}: {e}")
            faillist.append(ind)
            fail += 1
        finally:

            if success % 100 == 0:
                print(f"{success} / {total} (Failed: {fail})")

    df = pd.DataFrame(rows_list)

    
    os.makedirs(os.path.dirname(listspath + file), exist_ok=True)

    df.to_csv(listspath + file, index=False)
    print("Failed list:", faillist)
    print("Empty graphs:", sh)
    print(f"Saved to {listspath + file}")


if __name__ == "__main__":
    extract_cfglist("filteredcfg_merge_graph.csv")
    extract_cfglist("filteredcfg_nomerge_graph.csv")

    
    