

from parse.doc2vec import Doc2Vec, TaggedDocument
from parse.features import WeisfeilerLehmanHashing
from parse.estimator import Estimator

import numpy as np
import networkx as nx
from typing import List


class Cfg2Vec(Estimator):
    def __init__(
            self,
            wl_iterations: int = 2,
            attributed: bool = False,
            dimensions: int = 128,
            workers: int = 4,
            down_sampling: float = 0.0001,
            epochs: int = 10,
            learning_rate: float = 0.025,
            min_count: int = 1,
            seed: int = 42,
            erase_base_features: bool = True,
            use_wl: bool = True,
            use_path: bool = True
    ):
        self.wl_iterations = wl_iterations
        self.attributed = attributed
        self.dimensions = dimensions
        self.workers = workers
        self.down_sampling = down_sampling
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed
        self.erase_base_features = erase_base_features
        self.use_wl = use_wl
        self.use_path = use_path

    def fit(self, graphs: List[nx.classes.graph.Graph]):
        self._set_seed()
        graphs = self._check_graphs(graphs)

        documents = []
        for i, graph in enumerate(graphs):

            w = WeisfeilerLehmanHashing(
                graph,
                self.wl_iterations,
                self.attributed,
                self.erase_base_features,
                self.use_wl,
                self.use_path
            )
            documents.append(w)


        tagged_documents = [
            TaggedDocument(words=doc.get_graph_features(), tags=[str(i)])
            for i, doc in enumerate(documents) if doc.get_graph_features()
        ]

        if not tagged_documents:
            print("Warning: No graph features were generated. The embedding will be empty.")
            self._embedding = []

            self._embedding = np.zeros((len(graphs), self.dimensions))
            return


        original_indices = [i for i, doc in enumerate(documents) if doc.get_graph_features()]
        index_map = {original_idx: new_idx for new_idx, original_idx in enumerate(original_indices)}

        self.model = Doc2Vec(
            tagged_documents,
            vector_size=self.dimensions,
            window=0,
            min_count=self.min_count,
            dm=0,
            sample=self.down_sampling,
            workers=self.workers,
            epochs=self.epochs,
            alpha=self.learning_rate,
            seed=self.seed,
        )


        self._embedding = np.zeros((len(graphs), self.dimensions))
        for original_idx in original_indices:
            new_idx = index_map[original_idx]
            self._embedding[original_idx] = self.model.docvecs[str(original_idx)]

    def get_embedding(self) -> np.array:
        return np.array(self._embedding)

    def infer(self, graphs) -> np.array:
        self._set_seed()
        graphs = self._check_graphs(graphs)


        documents = [
            WeisfeilerLehmanHashing(
                graph,
                self.wl_iterations,
                self.attributed,
                self.erase_base_features,
                self.use_wl,
                self.use_path
            )
            for graph in graphs
        ]

        documents = [doc.get_graph_features() for _, doc in enumerate(documents)]

        embedding = np.array(
            [
                self.model.infer_vector(
                    doc, alpha=self.learning_rate, min_alpha=0.00001, epochs=self.epochs
                ) if doc else np.zeros(self.dimensions)
                for doc in documents
            ]
        )

        return embedding