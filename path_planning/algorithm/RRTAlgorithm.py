import random
import time

import numpy as np

from path_planning.algorithm.PathPlanningAlgorithm import PathPlanningAlgorithm
from path_planning.visualization.ProcessVisualizer import RRTProcessVisualizer
from path_planning.visualization.ResultVisualizer import ResultVisualizer


class RRTAlgorithm(PathPlanningAlgorithm):
    def __init__(self, max_iterations=3000, step_size=0.9, goal_sample_rate=0.3, max_extend_length=0.9,
                 show_process=True):
        """
        初始化RRT算法

        参数:
            max_iterations: 最大迭代次数
            step_size: 每次扩展的步长
            goal_sample_rate: 以目标点作为采样点的概率
            max_extend_length: 最大扩展长度
            visualize: 可视化
        """
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_extend_length = max_extend_length
        self.show_process = show_process
        self.visualize = RRTProcessVisualizer()




    def find_path(self, start, goal, planner):
        """
        使用RRT算法查找从起点到终点的路径，并可视化过程

        参数:
            start: 世界坐标中的起点
            goal: 世界坐标中的终点
            planner: PathPlanner实例，提供环境信息和辅助方法

        返回:
            路径点列表或None（如果未找到路径）
        """
        print("使用RRT算法寻找路径...")

        # 调整起点和终点，确保它们不在障碍物内
        start, _ = self.adjust_point_if_in_collision(start, planner)
        goal, _ = self.adjust_point_if_in_collision(goal, planner)

        # 初始化RRT树
        tree = {start: None}  # 节点: 父节点

        # 自适应调整步长
        self.step_size = min(planner.voxel_size * 3, self.step_size)
        print(f"RRT步长设置为: {self.step_size}")

        # 设置搜索边界
        bounds = planner.get_bounds()
        # 正确检查边界是否存在
        if bounds is not None and isinstance(bounds, tuple) and len(bounds) == 2:
            min_bounds = np.array(bounds[0])
            max_bounds = np.array(bounds[1])
        else:
            # 如果没有明确的边界，根据起点和终点创建一个搜索空间
            min_bounds = np.minimum(np.array(start), np.array(goal)) - 100
            max_bounds = np.maximum(np.array(start), np.array(goal)) + 100

        # 初始化可视化 - 传入planner以获取mesh
        self.visualize.show(planner.mesh, start, goal, None)

        # 开始RRT迭代
        for i in range(self.max_iterations):
            if i % 100 == 0:
                print(f"RRT迭代: {i}/{self.max_iterations}")

            # 随机采样一个点
            if random.random() < self.goal_sample_rate:
                random_point = goal
            else:
                random_point = self.random_point(min_bounds, max_bounds)

            # 找到树中最近的节点
            nearest_node = self.nearest_node(random_point, tree)

            # 从最近节点向随机点扩展
            new_node = self.extend(nearest_node, random_point, planner)

            if new_node:
                # 将新节点添加到树中
                tree[new_node] = nearest_node

                # 更新可视化
                if self.show_process:
                    self.visualize.update(tree, new_node, nearest_node)

                # 检查是否可以连接到目标
                if self.distance(new_node, goal) < self.max_extend_length:
                    if not self.is_path_collision(new_node, goal, planner):
                        tree[goal] = new_node
                        print(f"RRT找到路径，迭代次数: {i + 1}")
                        path = self.extract_path(tree, start, goal, planner)

                        # 最终路径可视化
                        if self.show_process:
                            self.visualize.update(tree, path=path)

                            # 保持窗口打开直到用户关闭
                            print("路径规划完成，按Q关闭可视化窗口...")
                            while True:
                                self.visualize.vis_o3d.poll_events()
                                self.visualize.vis_o3d.update_renderer()
                                time.sleep(0.1)
                                # 检查是否按下Q键或关闭窗口
                                if not self.visualize.vis_o3d.poll_events():
                                    break

                        return path

        print("RRT未能找到路径，达到最大迭代次数")

        # 如果没找到路径，仍然显示最终的RRT树
        if self.visualize:
            self.visualize.update(tree)
            # 保持窗口打开直到用户关闭
            print("未找到路径，按Q关闭可视化窗口...")
            while True:
                self.visualize.vis_o3d.poll_events()
                self.visualize.vis_o3d.update_renderer()
                time.sleep(0.1)
                # 检查是否按下Q键或关闭窗口
                if not self.visualize.vis_o3d.poll_events():
                    break
        return None

    def random_point(self, min_bounds, max_bounds):
        """生成随机点"""
        return tuple(random.uniform(min_bounds[i], max_bounds[i]) for i in range(3))

    def nearest_node(self, point, tree):
        """找到树中离给定点最近的节点"""
        return min(tree.keys(), key=lambda n: self.distance(n, point))

    def distance(self, p1, p2):
        """计算两点之间的欧氏距离"""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def extend(self, from_node, to_point, planner):
        """改进的扩展策略"""
        # 计算方向向量
        direction = np.array(to_point) - np.array(from_node)
        norm = np.linalg.norm(direction)

        if norm < 1e-6:  # 避免除零错误
            return None

        # 使用更小的步长
        actual_step_size = min(self.step_size, norm, planner.voxel_size * 0.8)
        direction = direction / norm * actual_step_size

        # 计算新节点
        new_point = tuple(np.array(from_node) + direction)

        # 更严格的检查
        if not planner.is_within_bounds(new_point):
            return None

        # 检查新点是否在障碍物内
        if self.is_collision(new_point, planner):
            return None

        # 检查路径是否有碰撞
        if self.is_path_collision(from_node, new_point, planner):
            return None

        return new_point

    def is_collision(self, point, planner):
        """改进的碰撞检测函数"""
        return planner.voxel_index.is_occupied(point)

    def is_path_collision(self, p1, p2, planner):
        """增强的路径碰撞检测"""
        # 计算两点之间的距离
        dist = self.distance(p1, p2)

        # 更密集的采样
        steps = max(10, int(dist / (planner.voxel_size * 0.25)))

        # 在路径上均匀采样点进行碰撞检测
        for i in range(1, steps):
            t = i / steps
            interpolated_point = tuple(np.array(p1) * (1 - t) + np.array(p2) * t)
            if self.is_collision(interpolated_point, planner):
                return True

        return False

    def extract_path(self, tree, start, goal, planner):
        """从树中提取路径并进行后处理"""
        path = [goal]
        current = goal

        while current != start:
            current = tree[current]
            path.append(current)

        path.reverse()

        # 路径后处理：确保所有点都远离障碍物
        processed_path = []
        for point in path:
            adjusted_point, was_adjusted = self.adjust_point_if_in_collision(point, planner)
            processed_path.append(adjusted_point)

        return processed_path

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

        # 如果简单调整不行，尝试更大范围的调整
        for distance in [2, 3, 4, 5]:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        offset = (dx * planner.voxel_size * distance,
                                  dy * planner.voxel_size * distance,
                                  dz * planner.voxel_size * distance)
                        new_point = tuple(np.array(point) + np.array(offset))
                        if not self.is_collision(new_point, planner):
                            print(f"点已调整为: {new_point}")
                            return new_point, True

        print(f"警告：无法调整点 {point} 使其不在障碍物内")
        return point, False