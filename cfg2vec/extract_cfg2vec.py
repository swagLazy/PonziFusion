import pandas as pd
import networkx as nx
import numpy as np
import os
import json
from parse.cfg2vec import Cfg2Vec

cfgpath = "../dataset/graphlists/filteredcfg_merge_graph.csv"
xiaorongcfgpath = "../dataset/graphlists/filteredcfg_nomerge_graph.csv"

vecfeatuespath = "../dataset/features/cfg2vec/"
ablationpath = "../dataset/features/ablation/"
os.makedirs(vecfeatuespath, exist_ok=True)
os.makedirs(ablationpath, exist_ok=True)


def generate_graph_embeddings(input_graph_file, output_feature_file, epoch, use_wl, use_path):
    print(f"--- Generating: {os.path.basename(output_feature_file)} ---")
    print(f"Params: epoch={epoch}, use_wl={use_wl}, use_path={use_path}")

    data = pd.read_csv(input_graph_file, index_col="address")
    print(f"Loaded {len(data)} graphs from {os.path.basename(input_graph_file)}")

    graphs = []
    address_map = {}

    for i, ind in enumerate(data.index):
        try:
            nodelist = json.loads(data.loc[ind, "nodes"])
            edgelist = json.loads(data.loc[ind, "edges"])
            g = nx.DiGraph()
            g.add_nodes_from(nodelist)
            g.add_edges_from(edgelist)
            if len(g) > 0:
                graphs.append(g)
                address_map[ind] = len(graphs) - 1
        except Exception as e:
            print(f"Warning: Failed to process graph for address {ind}. Error: {e}")

    print(f"Successfully constructed {len(graphs)} graph objects.")

    model = Cfg2Vec(dimensions=100, wl_iterations=epoch, use_wl=use_wl, use_path=use_path)
    model.fit(graphs)
    vecs = model.get_embedding()
    print("Model fitting finished.")

    veccol = ['vec_' + str(i) for i in range(100)]
    results = []
    for address, graph_index in address_map.items():
        embedding = vecs[graph_index].tolist()
        row = {'address': address}
        row.update(dict(zip(veccol, embedding)))
        results.append(row)

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_feature_file, index=False)
    print(f"Features saved to {output_feature_file}\n")


if __name__ == "__main__":
    # for epoch in range(1, 11):
    #     output_file = os.path.join(vecfeatuespath, f"filteredcfg_cfg2vec_{epoch}.csv")
    #     generate_graph_embeddings(
    #         input_graph_file=cfgpath,
    #         output_feature_file=output_file,
    #         epoch=epoch,
    #         use_wl=True,
    #         use_path=True
    #     )

    ablation_epoch = 5
    output_file_nomerge = os.path.join(ablationpath, f"xiaorong_cfg2vec_nomerge_{ablation_epoch}.csv")
    # generate_graph_embeddings(
    #     input_graph_file=xiaorongcfgpath,
    #     output_feature_file=output_file_nomerge,
    #     epoch=ablation_epoch,
    #     use_wl=True,
    #     use_path=True
    # )
    #
    output_file_subgraph = os.path.join(ablationpath, f"xiaorong_cfg2vec_nosubgraph_{ablation_epoch}.csv")
    generate_graph_embeddings(
        input_graph_file=cfgpath,
        output_feature_file=output_file_subgraph,
        epoch=ablation_epoch,
        use_wl=True,
        use_path=False
    )

    # output_file_path = os.path.join(ablationpath, f"xiaorong_cfg2vec_nopath_{ablation_epoch}.csv")
    # generate_graph_embeddings(
    #     input_graph_file=cfgpath,
    #     output_feature_file=output_file_path,
    #     epoch=ablation_epoch,
    #     use_wl=False,
    #     use_path=True
    # )