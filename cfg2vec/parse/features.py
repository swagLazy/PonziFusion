# 文件：features.py

import hashlib
import networkx as nx
from typing import List, Dict


class WeisfeilerLehmanHashing(object):
    """
    Weisfeiler-Lehman feature extractor class.

    Args:
        graph (NetworkX graph): NetworkX graph for which we do WL hashing.
        wl_iterations (int): Number of WL iterations.
        attributed (bool): Presence of attributes.
        erase_base_feature (bool): Deleting the base features.
        use_wl (bool): Flag to include Weisfeiler-Lehman subgraph features.
        use_path (bool): Flag to include path sequence features.
    """

    def __init__(
            self,
            graph: nx.classes.graph.Graph,
            wl_iterations: int,
            attributed: bool,
            erase_base_features: bool,
            use_wl: bool,
            use_path: bool
    ):
        """
        Initialization method which also executes feature extraction.
        """
        self.graph = graph
        self.wl_iterations = wl_iterations
        self.attributed = attributed
        self.erase_base_features = erase_base_features
        self.use_wl = use_wl
        self.use_path = use_path
        self.use_degree = True

        # Core logic: always calculate all features for on-demand combination later
        self._set_features()
        self._do_recursions()
        self._extract_path_features()

    def _set_features(self):
        """
        Creating the base features.
        """
        if self.attributed:
            self.features = nx.get_node_attributes(self.graph, "feature")
        else:
            self.features = {
                node: self.graph.degree(node) for node in self.graph.nodes()
            }
        self.extracted_features = {k: [str(v)] for k, v in self.features.items()}

    def _erase_base_features(self):
        """
        Erasing the base features
        """
        for k, v in self.extracted_features.items():
            del self.extracted_features[k][0]

    def _do_a_recursion(self):
        """
        The method does a single WL recursion.
        """
        new_features = {}
        for node in self.graph.nodes():
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])] + sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = {
            k: self.extracted_features[k] + [v] for k, v in new_features.items()
        }
        return new_features

    def _do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.wl_iterations):
            self.features = self._do_a_recursion()

        if self.erase_base_features:
            self._erase_base_features()

    def get_node_features(self) -> Dict[int, List[str]]:
        """
        Return the node level features.
        """
        return self.extracted_features

    def get_graph_features(self) -> List[str]:
        """
        Return the graph level features based on flags.
        """
        final_features = []

        if self.use_wl:
            wl_features = [
                feature
                for node, features in self.extracted_features.items()
                for feature in features
            ]
            final_features.extend(wl_features)

        if self.use_path:
            final_features.extend(self.path_features)

        if not final_features and self.extracted_features:
            base_features = [
                feature
                for node, features in self.extracted_features.items()
                for feature in features
            ]
            final_features.extend(base_features)

        return final_features

    def _dfs(self, node, visited, paths):
        """
        Corrected Depth-First Search to handle cycles.
        'visited' now tracks the nodes in the current path.
        """
        visited.append(node)

        # Base case: If the node is a leaf (no outgoing edges), we found a complete path.
        if self.graph.out_degree(node) == 0:
            paths.append(visited.copy())
        else:
            for i in self.graph.successors(node):
                # This is the crucial cycle check.
                # If the neighbor is already in our current path, skip it.
                if i in visited:
                    continue
                self._dfs(i, visited, paths)

        # Backtrack: Remove the node from the current path before returning.
        # This allows the node to be visited again as part of a different path.
        visited.pop()

    def _extract_path_features(self):
        if len(self.graph) == 0:
            self.path_features = []
            return

        paths = []
        # Iterate through all nodes that could be a starting point (in-degree is 0).
        for start_node in self.graph.nodes:
            if self.graph.in_degree(start_node) == 0:
                self._dfs(start_node, [], paths)

        # If no paths were found (e.g., a graph with only cycles),
        # start from an arbitrary node to get at least some path info.
        if not paths and len(self.graph.nodes) > 0:
            node = list(self.graph.nodes)[0]
            self._dfs(node, [], paths)

        if self.use_degree:
            paths = self._change_to_degree(paths)

        features = []
        for i in paths:
            p = [str(num) for num in i]
            f = "_".join(p)
            hash_object = hashlib.md5(f.encode())
            hashing = hash_object.hexdigest()
            features.append(hashing)
        self.path_features = features

    def _change_to_degree(self, paths):
        res = []
        for path in paths:
            newpath = []
            for node in path:
                newpath.append(self.graph.degree(node))
            res.append(newpath)
        return res