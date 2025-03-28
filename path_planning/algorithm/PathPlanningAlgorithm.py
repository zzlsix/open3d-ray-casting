from abc import ABC, abstractmethod


# 抽象路径规划算法接口
class PathPlanningAlgorithm(ABC):
    @abstractmethod
    def find_path(self, start_grid, goal_grid, planner):
        """
        查找从起点到终点的路径

        参数:
            start_grid: 网格坐标中的起点
            goal_grid: 网格坐标中的终点
            planner: PathPlanner实例，提供环境信息和辅助方法

        返回:
            路径点列表或None（如果未找到路径）
        """
        pass
