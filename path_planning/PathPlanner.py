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

    def is_collision(self, point):
        """检查点是否与障碍物碰撞"""
        return GeometryTools.is_point_in_voxels(point, self.occupied_voxels, self.voxel_size)

    def adjust_point_if_in_collision(self, point):
        """如果点在障碍物内，尝试调整"""
        if not self.is_collision(point):
            return point, False

        print(f"警告：点在障碍物内，尝试小幅调整...")
        for offset in [
            (self.voxel_size, 0, 0), (-self.voxel_size, 0, 0),
            (0, self.voxel_size, 0), (0, -self.voxel_size, 0),
            (0, 0, self.voxel_size), (0, 0, -self.voxel_size)
        ]:
            new_point = tuple(np.array(point) + np.array(offset))
            if not self.is_collision(new_point):
                print(f"点已调整为: {new_point}")
                return new_point, True

        return point, False

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

    def post_process_path(self, path, original_start, original_end, offset_distance=0.1):
        """将路径点投影到表面附近"""
        if path is None:
            return None

        processed_path = []

        # 保留原始起点和终点
        processed_path.append(original_start)

        # 处理中间点 - 将它们投影到距离表面更近的位置
        for i in range(1, len(path) - 1):
            point = path[i]

            # 计算点到网格的最近点
            closest_point, distance, triangle_id = trimesh.proximity.closest_point(self.mesh, [point])
            closest_point = closest_point[0]

            # 计算表面法线
            normal = self.mesh.face_normals[triangle_id[0]]

            # 将点移动到表面附近但不在表面上
            # 小偏移，让路径非常接近表面
            projected_point = closest_point + normal * offset_distance

            processed_path.append(projected_point)

        # 添加原始终点
        processed_path.append(original_end)

        return processed_path
