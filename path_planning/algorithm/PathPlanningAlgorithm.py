from abc import ABC, abstractmethod

import numpy as np


# 抽象路径规划算法接口
class PathPlanningAlgorithm(ABC):
    @abstractmethod
    def find_path(self, start, goal, planner):
        """
        查找从起点到终点的路径

        参数:
            start: 世界坐标中的起点
            goal: 世界坐标中的终点
            planner: PathPlanner实例，提供环境信息和辅助方法

        返回:
            路径点列表或None（如果未找到路径）
        """
        pass

    def adjust_point_if_in_collision(self, point, planner):
        """如果点在障碍物内，尝试调整"""
        if not self.is_collision(point, planner):
            return point, False

        print(f"警告：点在障碍物内，尝试小幅调整...")
        for offset in [
            (planner.voxel_size, 0, 0), (-planner.voxel_size, 0, 0),
            (0, planner.voxel_size, 0), (0, -planner.voxel_size, 0),
            (0, 0, planner.voxel_size), (0, 0, -planner.voxel_size)
        ]:
            new_point = tuple(np.array(point) + np.array(offset))
            if not self.is_collision(new_point, planner):
                print(f"点已调整为: {new_point}")
                return new_point, True

        return point, False

    def is_collision(self, point, planner):
        pass

    def post_process_path(self, mesh, path, original_start, original_end, offset_distance=0.1):
        pass

