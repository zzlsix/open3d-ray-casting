import networkx as nx
import numpy as np

from path_planning.algorithm.PathPlanningAlgorithm import PathPlanningAlgorithm


# A*算法实现
class AStarAlgorithm(PathPlanningAlgorithm):
    def find_path(self, graph, start_idx, end_idx):
        # 定义启发式函数
        def heuristic(a, b):
            pos_a = graph.nodes[a]['position']
            pos_b = graph.nodes[b]['position']
            return np.linalg.norm(pos_a - pos_b)

        if not nx.has_path(graph, start_idx, end_idx):
            raise nx.NetworkXNoPath("起点和终点之间没有路径")

        path_indices = nx.astar_path(graph, start_idx, end_idx, heuristic)
        return path_indices