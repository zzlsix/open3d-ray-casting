from abc import ABC, abstractmethod

# 定义路径规划算法接口
class PathPlanningAlgorithm(ABC):
    @abstractmethod
    def find_path(self, graph, start_idx, end_idx):
        """
        查找从起点到终点的路径

        参数:
        graph - 用于规划的图
        start_idx - 起点索引
        end_idx - 终点索引

        返回:
        path_indices - 路径上的顶点索引列表
        """
        pass