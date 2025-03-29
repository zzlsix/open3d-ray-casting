import time
import numpy as np
import trimesh

from path_planning.GeometryTools import GeometryTools
from path_planning.algorithm.AStarAlgorithm import AStarAlgorithm


# 路径规划模块
class PathPlanner:
    def __init__(self, mesh, occupied_voxels, voxel_size, bounds=None, algorithm=AStarAlgorithm()):
        self.mesh = mesh
        self.occupied_voxels = occupied_voxels
        self.voxel_size = voxel_size
        self.bounds = bounds
        self.geometry_tools = GeometryTools()
        # 默认使用A*算法
        self.algorithm = algorithm

    def get_bounds(self):
        return self.bounds

    def set_algorithm(self, algorithm):
        """设置路径规划算法"""
        self.algorithm = algorithm

    def to_grid(self, point):
        """将连续坐标转换为离散网格坐标"""
        return tuple(np.round(np.array(point) / self.voxel_size).astype(int))

    def to_world(self, grid_point):
        """将离散网格坐标转换为连续坐标"""
        return tuple(np.array(grid_point) * self.voxel_size)

    def is_within_bounds(self, point):
        """检查点是否在边界内"""
        if self.bounds is None:
            return True
        # 增加一点余量，因为点可能在边界外一点点
        margin = self.voxel_size * 2
        return all(self.bounds[0][i] - margin <= point[i] <= self.bounds[1][i] + margin for i in range(3))

    def find_path(self, start, goal):
        """使用当前设置的算法寻找从起点到终点的路径"""
        # 使用选定的算法查找路径
        start_time = time.time()
        path = self.algorithm.find_path(start, goal, self)
        end_time = time.time()
        print(f"找到路径！用时: {end_time - start_time:.2f}秒")

        # 如果找到路径，添加真正的目标点
        if path:
            path.append(goal)
        return path


