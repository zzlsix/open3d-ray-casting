import random
import numpy as np

from math import sqrt

from path_planning.algorithm.PathPlanningAlgorithm import PathPlanningAlgorithm


# RRT算法实现
class RRTAlgorithm(PathPlanningAlgorithm):
    def __init__(self, max_iterations=20000, step_size=0.5, goal_sample_rate=0.2, search_radius=5.0):
        """初始化RRT算法参数"""
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius  # 用于局部采样

    def find_path(self, start_grid, goal_grid, planner):
        """使用RRT算法查找路径"""
        print(f"开始RRT搜索，从网格坐标 {start_grid} 到 {goal_grid}")

        # 转换为世界坐标
        start_world = planner.to_world(start_grid)
        goal_world = planner.to_world(goal_grid)

        # 计算起点和终点之间的距离，用于自适应参数调整
        start_to_goal_dist = np.linalg.norm(np.array(start_world) - np.array(goal_world))
        print(f"起点到终点的直线距离: {start_to_goal_dist}")

        # 根据起终点距离自适应调整步长
        if start_to_goal_dist < 10:
            self.step_size = min(0.3, start_to_goal_dist / 10)
            print(f"短距离路径规划，调整步长为: {self.step_size}")

        # RRT算法初始化
        tree = {}  # {node: parent_node}
        tree[start_grid] = None  # 根节点没有父节点

        # 使用两个树（双向RRT）可以加速收敛
        forward_tree = {start_grid: None}  # 从起点出发的树
        backward_tree = {goal_grid: None}  # 从终点出发的树
        trees = [forward_tree, backward_tree]
        current_tree_idx = 0  # 当前使用的树索引

        successful_iterations = 0  # 记录成功扩展的迭代次数

        for i in range(self.max_iterations):
            # 定期显示进度
            if i % 500 == 0:
                print(f"RRT迭代次数: {i}/{self.max_iterations}, 成功扩展: {successful_iterations}")

            # 切换树
            current_tree = trees[current_tree_idx]
            other_tree = trees[1 - current_tree_idx]

            # 随机采样或朝目标采样
            if random.random() < self.goal_sample_rate:
                # 目标引导采样：选择另一棵树的随机节点作为目标
                other_nodes = list(other_tree.keys())
                if other_nodes:
                    random_grid = random.choice(other_nodes)
                else:
                    random_grid = goal_grid if current_tree_idx == 0 else start_grid
            else:
                # 采用混合采样策略
                if random.random() < 0.7:  # 70%的概率使用局部采样
                    # 从当前树中选择一个随机节点
                    tree_nodes = list(current_tree.keys())
                    if tree_nodes:
                        base_node = random.choice(tree_nodes)
                        # 在该节点附近进行局部采样
                        random_grid = self._local_sample(base_node, planner)
                    else:
                        random_grid = self._random_state(planner)
                else:  # 30%的概率使用全局采样
                    random_grid = self._random_state(planner)

            random_world = planner.to_world(random_grid)

            # 确保采样点在有效范围内且无碰撞
            attempts = 0
            while (not planner.is_within_bounds(random_world) or
                   planner.is_collision(random_world)) and attempts < 20:
                if random.random() < 0.5:  # 混合采样策略
                    random_grid = self._local_sample(
                        random.choice(list(current_tree.keys())),
                        planner
                    )
                else:
                    random_grid = self._random_state(planner)
                random_world = planner.to_world(random_grid)
                attempts += 1

            if attempts >= 20:
                continue  # 放弃当前迭代，尝试下一次

            # 找到当前树中离随机点最近的节点
            nearest_grid = self._nearest_neighbor(random_grid, current_tree)

            # 从最近节点向随机点延伸一步
            new_grid = self._steer(nearest_grid, random_grid)
            new_world = planner.to_world(new_grid)

            # 检查新节点是否有效且路径无碰撞
            if (planner.is_within_bounds(new_world) and
                    not planner.is_collision(new_world) and
                    self._check_path(nearest_grid, new_grid, planner)):

                # 将新节点添加到当前树中
                current_tree[new_grid] = nearest_grid
                successful_iterations += 1

                # 查找另一棵树中离新节点最近的节点
                nearest_in_other = self._nearest_neighbor(new_grid, other_tree)

                # 尝试连接两棵树
                if nearest_in_other and self._distance(new_grid, nearest_in_other) < self.step_size * 3:
                    if self._check_path(new_grid, nearest_in_other, planner):
                        print("找到路径！两棵树成功连接")

                        # 构建路径
                        path = self._construct_path(new_grid, nearest_in_other, forward_tree, backward_tree,
                                                    current_tree_idx, planner)

                        print(f"路径长度: {len(path)}个点")
                        print(f"RRT迭代次数: {i + 1}, 成功扩展: {successful_iterations}")
                        return path

            # 交替使用两棵树
            current_tree_idx = 1 - current_tree_idx

        # 如果没有找到路径
        print(f"RRT迭代次数: {self.max_iterations}, 成功扩展: {successful_iterations}")
        print(f"没有找到路径")
        return None

    def _construct_path(self, node1, node2, forward_tree, backward_tree, current_tree_idx, planner):
        """构建完整路径"""
        if current_tree_idx == 0:  # 当前是前向树
            forward_node = node1
            backward_node = node2
        else:  # 当前是后向树
            forward_node = node2
            backward_node = node1

        # 构建前向路径
        forward_path = []
        current = forward_node
        while current is not None:
            forward_path.append(planner.to_world(current))
            current = forward_tree.get(current)
        forward_path.reverse()  # 前向路径需要反转

        # 构建后向路径
        backward_path = []
        current = backward_node
        while current is not None:
            backward_path.append(planner.to_world(current))
            current = backward_tree.get(current)

        # 合并路径
        return forward_path + backward_path

    def _local_sample(self, base_grid, planner):
        """在基准点附近进行局部采样"""
        # 在搜索半径内随机采样
        dx = random.uniform(-self.search_radius, self.search_radius)
        dy = random.uniform(-self.search_radius, self.search_radius)
        dz = random.uniform(-self.search_radius, self.search_radius)

        x = int(base_grid[0] + dx)
        y = int(base_grid[1] + dy)
        z = int(base_grid[2] + dz)

        return (x, y, z)

    def _random_state(self, planner):
        """生成随机的网格坐标"""
        # 获取环境边界
        bounds = planner.get_bounds()

        if bounds is None:
            # 如果没有明确的边界，使用一个合理的默认范围
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            z = random.randint(-100, 100)
        else:
            # 将世界坐标边界转换为网格坐标
            min_bound = planner.to_grid(bounds[0])
            max_bound = planner.to_grid(bounds[1])

            # 随机采样
            x = random.randint(min_bound[0], max_bound[0])
            y = random.randint(min_bound[1], max_bound[1])
            z = random.randint(min_bound[2], max_bound[2])

        return (x, y, z)

    def _nearest_neighbor(self, target_grid, tree):
        """找到树中离目标点最近的节点"""
        min_dist = float('inf')
        nearest = None

        for node in tree.keys():
            dist = self._distance(node, target_grid)
            if dist < min_dist:
                min_dist = dist
                nearest = node

        return nearest

    def _distance(self, grid1, grid2):
        """计算两个网格点之间的欧几里得距离"""
        return sqrt((grid1[0] - grid2[0]) ** 2 +
                    (grid1[1] - grid2[1]) ** 2 +
                    (grid1[2] - grid2[2]) ** 2)

    def _steer(self, from_grid, to_grid):
        """从起点向目标点延伸一定距离"""
        dist = self._distance(from_grid, to_grid)

        if dist <= self.step_size:
            return to_grid

        # 计算单位向量
        dx = (to_grid[0] - from_grid[0]) / dist
        dy = (to_grid[1] - from_grid[1]) / dist
        dz = (to_grid[2] - from_grid[2]) / dist

        # 沿该方向延伸step_size距离
        new_x = int(from_grid[0] + dx * self.step_size)
        new_y = int(from_grid[1] + dy * self.step_size)
        new_z = int(from_grid[2] + dz * self.step_size)

        return (new_x, new_y, new_z)

    def _check_path(self, from_grid, to_grid, planner):
        """检查两点之间的路径是否无碰撞"""
        # 获取两点间的一系列采样点
        points = self._interpolate(from_grid, to_grid)

        # 检查每个采样点是否无碰撞
        for grid in points:
            world = planner.to_world(grid)
            if not planner.is_within_bounds(world) or planner.is_collision(world):
                return False

        return True

    def _interpolate(self, from_grid, to_grid, num_points=10):
        """在两点之间进行线性插值以检查路径"""
        points = []
        for i in range(num_points + 1):
            t = i / num_points
            x = int(from_grid[0] + t * (to_grid[0] - from_grid[0]))
            y = int(from_grid[1] + t * (to_grid[1] - from_grid[1]))
            z = int(from_grid[2] + t * (to_grid[2] - from_grid[2]))
            points.append((x, y, z))

        return points
