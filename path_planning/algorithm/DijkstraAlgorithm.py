import networkx as nx

from path_planning.algorithm.PathPlanningAlgorithm import PathPlanningAlgorithm


# Dijkstra算法实现
class DijkstraAlgorithm(PathPlanningAlgorithm):
    def find_path(self, graph, start_idx, end_idx):
        if not nx.has_path(graph, start_idx, end_idx):
            raise nx.NetworkXNoPath("起点和终点之间没有路径")

        path_indices = nx.dijkstra_path(graph, start_idx, end_idx, weight='weight')
        return path_indices