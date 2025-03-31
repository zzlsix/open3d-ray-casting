import random

import numpy as np

from path_planning.algorithm.PathPlanningAlgorithm import PathPlanningAlgorithm


class RRTStarAlgorithm(PathPlanningAlgorithm):
    def __init__(self, max_iterations=10000, step_size=1.0, goal_sample_rate=0.20, search_radius=5.0):
        """
        初始化RRT*算法

        参数:
            max_iterations: 最大迭代次数
            step_size: 每次扩展的步长
            goal_sample_rate: 直接采样目标点的概率
            search_radius: 重新连接和优化的搜索半径
        """
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius

    def find_path(self, start, goal, planner):
        """
        使用RRT*算法查找从起点到终点的路径

        参数:
            start: 起点坐标
            goal: 终点坐标
            planner: PathPlanner实例

        返回:
            路径点列表或None（如果未找到路径）
        """
        print("使用RRT*算法寻找路径...")

        # 调整起点和终点，确保它们不在障碍物内
        start, adjusted_start = self.adjust_point_if_in_collision(start, planner)
        goal, adjusted_goal = self.adjust_point_if_in_collision(goal, planner)

        if adjusted_start:
            print(f"起点已调整为: {start}")
        if adjusted_goal:
            print(f"终点已调整为: {goal}")

        # 检查起点和终点是否在边界内
        if not planner.is_within_bounds(start) or not planner.is_within_bounds(goal):
            print("起点或终点超出边界！")
            return None

        # 初始化树
        nodes = [start]  # 节点列表
        parents = [-1]  # 每个节点的父节点索引
        costs = [0.0]  # 从起点到每个节点的成本

        # 主循环
        for i in range(self.max_iterations):
            # 随机采样
            if random.random() < self.goal_sample_rate:
                random_point = goal
            else:
                random_point = self.random_sample(planner)

            # 找到最近的节点
            nearest_idx = self.find_nearest(nodes, random_point)
            nearest_node = nodes[nearest_idx]

            # 按步长扩展
            new_node = self.steer(nearest_node, random_point, self.step_size)

            # 检查新节点是否有效
            if not self.is_collision(new_node, planner) and planner.is_within_bounds(new_node):
                # 找到附近的节点
                near_indices = self.find_near_nodes(nodes, new_node, self.search_radius)

                # 选择成本最低的父节点
                min_cost = costs[nearest_idx] + self.distance(nearest_node, new_node)
                min_parent = nearest_idx

                for near_idx in near_indices:
                    near_node = nodes[near_idx]
                    if not self.is_collision_free(near_node, new_node, planner):
                        continue

                    cost = costs[near_idx] + self.distance(near_node, new_node)
                    if cost < min_cost:
                        min_cost = cost
                        min_parent = near_idx

                # 添加新节点到树中
                nodes.append(new_node)
                parents.append(min_parent)
                costs.append(min_cost)
                new_idx = len(nodes) - 1

                # 重新连接 - 检查是否可以通过新节点降低附近节点的成本
                self.rewire(nodes, parents, costs, new_idx, near_indices, planner)

                # 检查是否可以连接到目标
                if self.distance(new_node, goal) < self.step_size and not self.is_collision_free(new_node, goal,
                                                                                                 planner):
                    # 找到路径
                    nodes.append(goal)
                    parents.append(new_idx)
                    costs.append(costs[new_idx] + self.distance(new_node, goal))

                    # 构建路径
                    path = self.extract_path(nodes, parents)
                    print(f"RRT*找到路径，迭代次数: {i + 1}")
                    return path

            # 每100次迭代打印进度
            if (i + 1) % 100 == 0:
                print(f"RRT*迭代: {i + 1}/{self.max_iterations}, 节点数: {len(nodes)}")

        # 尝试连接到最近的节点
        nearest_to_goal = self.find_nearest(nodes, goal)
        if self.distance(nodes[nearest_to_goal], goal) < self.step_size * 2 and self.is_collision_free(
                nodes[nearest_to_goal], goal, planner):
            nodes.append(goal)
            parents.append(nearest_to_goal)
            path = self.extract_path(nodes, parents)
            print("RRT*找到近似路径")
            return path

        print("RRT*未能找到路径")
        return None

    def random_sample(self, planner):
        """生成随机采样点"""
        bounds = planner.get_bounds()
        if bounds is None:
            # 如果没有明确的边界，在起点周围采样
            center = np.array([0, 0, 0])
            radius = 100.0

            # 随机方向
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)

            # 随机距离
            distance = radius * random.random()

            return tuple(center + direction * distance)
        else:
            # 在边界内随机采样
            min_bound, max_bound = bounds
            x = random.uniform(min_bound[0], max_bound[0])
            y = random.uniform(min_bound[1], max_bound[1])
            z = random.uniform(min_bound[2], max_bound[2])
            return (x, y, z)

    def find_nearest(self, nodes, point):
        """找到最近的节点索引"""
        distances = [self.distance(node, point) for node in nodes]
        return distances.index(min(distances))

    def steer(self, from_node, to_node, step_size):
        """按步长从一个节点向另一个节点扩展"""
        dist = self.distance(from_node, to_node)
        if dist <= step_size:
            return to_node

        # 计算单位向量
        direction = np.array(to_node) - np.array(from_node)
        direction = direction / np.linalg.norm(direction)

        # 按步长扩展
        new_point = np.array(from_node) + direction * step_size
        return tuple(new_point)

    def distance(self, point1, point2):
        """计算两点之间的欧几里得距离"""
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def is_collision(self, point, planner):
        """检查点是否与障碍物碰撞"""
        # 使用体素网格检查碰撞
        return planner.voxel_index.is_occupied(point)

    def is_collision_free(self, point1, point2, planner):
        """检查两点之间的路径是否无碰撞"""
        dist = self.distance(point1, point2)

        # 如果距离很小，直接返回
        if dist < 0.01:
            return True

        # 沿着路径检查多个点
        num_checks = max(3, int(dist / (planner.voxel_size / 2)))
        for i in range(1, num_checks):
            t = i / num_checks
            check_point = tuple(np.array(point1) * (1 - t) + np.array(point2) * t)
            if self.is_collision(check_point, planner):
                return False

        return True

    def find_near_nodes(self, nodes, point, radius):
        """找到给定半径内的所有节点索引"""
        indices = []
        for i, node in enumerate(nodes):
            if self.distance(node, point) <= radius:
                indices.append(i)
        return indices

    def rewire(self, nodes, parents, costs, new_idx, near_indices, planner):
        """重新连接 - 尝试通过新节点优化附近节点的路径"""
        new_node = nodes[new_idx]

        for near_idx in near_indices:
            near_node = nodes[near_idx]

            # 计算通过新节点的成本
            potential_cost = costs[new_idx] + self.distance(new_node, near_node)

            # 如果成本更低且路径无碰撞，则重新连接
            if potential_cost < costs[near_idx] and self.is_collision_free(new_node, near_node, planner):
                parents[near_idx] = new_idx
                costs[near_idx] = potential_cost

    def extract_path(self, nodes, parents):
        """从树中提取路径"""
        path = []
        current_idx = len(nodes) - 1  # 目标节点的索引

        while current_idx != -1:
            path.append(nodes[current_idx])
            current_idx = parents[current_idx]

        path.reverse()  # 反转路径，使其从起点到终点
        return path