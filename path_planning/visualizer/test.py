import networkx as nx
import numpy as np
import trimesh
from scipy.spatial import KDTree

from path_planning.algorithm.AStarAlgorithm import AStarAlgorithm
from path_planning.checker.CollisionChecker import CollisionChecker
from path_planning.smoother.PathSmoother import PathSmoother
from path_planning.visualizer.PathVisualizer import PathVisualizer


# 主要的路径规划类
class PathPlanner:
    def __init__(self, algorithm=None):
        """
        初始化路径规划器

        参数:
        algorithm - 用于规划路径的算法，默认为A*算法
        """
        self.algorithm = algorithm if algorithm else AStarAlgorithm()
        self.mesh = None
        self.graph = None
        self.collision_checker = None
        self.path_smoother = None
        self.visualizer = PathVisualizer()

    def load_mesh(self, mesh_path):
        """加载3D模型网格"""
        print("加载3D模型...")
        loaded_obj = trimesh.load(mesh_path)

        # 处理可能是Scene的情况
        if isinstance(loaded_obj, trimesh.Scene):
            print("加载的是一个场景，尝试提取网格...")
            # 提取所有几何体
            geometries = list(loaded_obj.geometry.values())
            if len(geometries) == 0:
                raise ValueError("场景中没有找到网格")

            # 如果有多个网格，将它们合并为一个
            if len(geometries) > 1:
                print(f"合并场景中的 {len(geometries)} 个网格")
                self.mesh = trimesh.util.concatenate(geometries)
            else:
                self.mesh = geometries[0]
        else:
            self.mesh = loaded_obj

            # 检查模型属性
        if hasattr(self.mesh, 'is_watertight') and not self.mesh.is_watertight:
            print("警告: 模型不是水密的，可能会影响路径规划质量")

            # 初始化碰撞检测器和路径平滑器
        self.collision_checker = CollisionChecker(self.mesh)
        self.path_smoother = PathSmoother(self.mesh, self.collision_checker)

        return self.mesh

    def build_graph(self):
        """根据网格构建图形拓扑"""
        if self.mesh is None:
            raise ValueError("请先加载网格")

        print("构建网络图...")
        self.graph = nx.Graph()
        vertices = self.mesh.vertices

        # 添加所有顶点
        for i, vertex in enumerate(vertices):
            self.graph.add_node(i, position=vertex)

        # 添加所有边，使用网格的边连接信息
        if hasattr(self.mesh, 'edges_unique'):
            edges = self.mesh.edges_unique
        else:
            # 如果没有edges_unique属性，尝试从面构建边
            edges = set()
            for face in self.mesh.faces:
                for i in range(len(face)):
                    edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
                    edges.add(edge)
            edges = list(edges)

        for edge in edges:
            v1, v2 = edge
            # 计算边的长度作为权重
            weight = np.linalg.norm(vertices[v1] - vertices[v2])
            self.graph.add_edge(v1, v2, weight=weight)

        return self.graph

    def find_nearest_vertices(self, point, k=5):
        """找到网格上最接近给定点的顶点"""
        if self.mesh is None:
            raise ValueError("请先加载网格")

        vertices = self.mesh.vertices
        kdtree = KDTree(vertices)

        # 查找多个最近点而不只是一个，以防最近点不在主要连通分量上
        dists, indices = kdtree.query(point, k=k)

        return dists, indices

    def connect_components(self, max_connection_distance=10.0, n_connections=3):
        """连接图中的不连通分量"""
        if self.graph is None:
            raise ValueError("请先构建图")

        print("分析图的连通性...")
        connected_components = list(nx.connected_components(self.graph))
        print(f"发现 {len(connected_components)} 个连通分量")

        if len(connected_components) <= 1:
            return self.graph  # 已经是连通的

        vertices = self.mesh.vertices

        # 为每个组件计算一个中心点
        component_centers = []
        for component in connected_components:
            # 使用numpy向量化操作计算中心点
            comp_vertices_indices = np.array(list(component))
            comp_vertices = vertices[comp_vertices_indices]
            center = np.mean(comp_vertices, axis=0)
            component_centers.append((center, list(component)[0]))  # 中心和一个代表点

        # 使用numpy向量化操作计算距离矩阵
        centers_array = np.array([center for center, _ in component_centers])
        reps_array = np.array([rep for _, rep in component_centers])

        # 一次性计算所有组件对之间的距离
        for i in range(len(component_centers)):
            # 向量化计算当前中心到所有其他中心的距离
            center_i = centers_array[i]
            diffs = centers_array - center_i
            distances = np.linalg.norm(diffs, axis=1)

            # 创建索引、距离和代表点的数组
            indices = np.arange(len(component_centers))
            valid_indices = indices != i  # 排除自身

            valid_distances = distances[valid_indices]
            valid_j_indices = indices[valid_indices]
            valid_reps = reps_array[valid_indices]

            # 找出最近的N个组件
            if len(valid_distances) > 0:
                nearest_indices = np.argsort(valid_distances)[:n_connections]
                for idx in nearest_indices:
                    d = valid_distances[idx]
                    j = valid_j_indices[idx]
                    rep_j = valid_reps[idx]

                    if d <= max_connection_distance:
                        print(f"连接组件 {i} 到 {j}, 距离: {d:.2f}")
                        self.graph.add_edge(component_centers[i][1], rep_j, weight=d)

        return self.graph

    def plan_path(self, start_point, end_point,
                  collision_safety_margin=0.01,
                  connect_components=True,
                  max_connection_distance=10.0,
                  visualize=True):
        """
        在3D模型上规划两点之间的路径

        参数:
        start_point - 起始点坐标 [x, y, z]
        end_point - 终点坐标 [x, y, z]
        collision_safety_margin - 碰撞检测的安全距离
        connect_components - 是否尝试连接不连通的组件
        max_connection_distance - 最大连接距离
        visualize - 是否可视化路径

        返回:
        final_path - 最终规划的路径点列表
        """
        if self.mesh is None:
            raise ValueError("请先加载网格")

        if self.graph is None:
            self.build_graph()

        # 更新碰撞检测器的安全距离
        self.collision_checker.safety_margin = collision_safety_margin

        # 1. 找到最接近起点和终点的顶点
        print("查找模型上的起点和终点...")
        vertices = self.mesh.vertices
        start_dists, start_indices = self.find_nearest_vertices(start_point)
        end_dists, end_indices = self.find_nearest_vertices(end_point)

        # 初始设为第一个最近点
        start_idx = start_indices[0]
        end_idx = end_indices[0]

        start_vertex = vertices[start_idx]
        end_vertex = vertices[end_idx]

        print(f"最接近的起点: {start_vertex}, 索引: {start_idx}")
        print(f"最接近的终点: {end_vertex}, 索引: {end_idx}")

        # 2. 分析连通性
        connected_components = list(nx.connected_components(self.graph))

        # 找出起点和终点所在的连通分量
        start_component = None
        end_component = None

        for i, component in enumerate(connected_components):
            if start_idx in component:
                start_component = i
            if end_idx in component:
                end_component = i

        print(f"起点在连通分量 {start_component}，终点在连通分量 {end_component}")

        # 3. 处理连通性问题
        if start_component != end_component and start_component is not None and end_component is not None:
            print("起点和终点不在同一连通分量中")

            if connect_components:
                print("尝试连接不同的连通分量...")

                # 预先为每个顶点创建一个组件索引映射，避免重复循环
                vertex_to_component = {}
                for i, component in enumerate(connected_components):
                    for vertex_idx in component:
                        vertex_to_component[vertex_idx] = i

                # 策略1: 尝试不同的起点和终点候选
                found_connection = False
                for s_idx in start_indices:
                    s_comp = vertex_to_component.get(s_idx)
                    if s_comp is None:
                        continue

                    for e_idx in end_indices:
                        e_comp = vertex_to_component.get(e_idx)
                        if e_comp is None:
                            continue

                        if s_comp == e_comp:
                            print(f"找到在同一连通分量的替代起点 {s_idx} 和终点 {e_idx}")
                            start_idx = s_idx
                            end_idx = e_idx
                            start_vertex = vertices[start_idx]
                            end_vertex = vertices[end_idx]
                            found_connection = True
                            break

                    if found_connection:
                        break

                # 策略2: 如果无法找到同一分量的点，则添加连接
                if not found_connection:
                    self.connect_components(max_connection_distance)

                    # 检查起点和终点现在是否连通
                    if not nx.has_path(self.graph, start_idx, end_idx):
                        # 最后的尝试：直接连接起点和终点
                        direct_distance = np.linalg.norm(start_vertex - end_vertex)
                        if direct_distance <= max_connection_distance:
                            print(f"直接连接起点和终点，距离: {direct_distance:.2f}")
                            self.graph.add_edge(start_idx, end_idx, weight=direct_distance)

        # 4. 使用所选算法找到路径
        print(f"使用{self.algorithm.__class__.__name__}规划路径...")
        try:
            path_indices = self.algorithm.find_path(self.graph, start_idx, end_idx)
            print(f"找到路径，长度: {len(path_indices)}个点")

            # 提取路径的顶点坐标
            path_vertices = [vertices[idx] for idx in path_indices]

        except nx.NetworkXNoPath:
            print("无法找到路径，可能需要调整连接参数或选择不同的起点和终点")

            if visualize:
                # 尽管没有路径，但仍然可视化网格和两点
                self.visualizer.visualize_with_open3d(self.mesh, start_vertex, end_vertex, None)
            return None

        # 5. 碰撞检测
        print("进行碰撞检测...")
        collision_points = []

        # for i in range(len(path_indices) - 1):
        #     v1 = vertices[path_indices[i]]
        #     v2 = vertices[path_indices[i + 1]]
        #
        #     # 使用更精确的碰撞检测
        #     if self.collision_checker.ray_collision_check(v1, v2):
        #         print(f"检测到碰撞在路径段 {i} 到 {i + 1}")
        #         collision_points.append((i, v1, v2))
        #
        # if collision_points:
        #     print(f"发现 {len(collision_points)} 个碰撞点")
        #
        #     # 处理碰撞点 - 简单方法是从路径中移除碰撞段
        #     if len(collision_points) < len(path_indices) - 1:  # 确保不会移除所有路径
        #         new_path_indices = []
        #         skip_mode = False
        #
        #         for i in range(len(path_indices)):
        #             if i < len(path_indices) - 1:
        #                 # 检查当前边是否是碰撞边
        #                 is_collision_edge = any(cp[0] == i for cp in collision_points)
        #
        #                 if is_collision_edge:
        #                     skip_mode = True
        #                 elif skip_mode:
        #                     skip_mode = False
        #                     new_path_indices.append(path_indices[i])
        #                 else:
        #                     new_path_indices.append(path_indices[i])
        #             else:
        #                 # 总是保留最后一个点
        #                 new_path_indices.append(path_indices[i])
        #
        #         path_indices = new_path_indices
        #         print(f"移除碰撞段后的路径长度: {len(path_indices)}")
        #
        #         # 重新提取路径顶点
        #         path_vertices = [vertices[idx] for idx in path_indices]
        #     else:
        #         print("警告: 所有路径段都存在碰撞，无法简单修复")
        # else:
        #     print("路径无碰撞")

        # 6. 平滑路径
        if len(path_indices) > 2:
            print("平滑路径...")
            final_path = path_vertices # self.path_smoother.smooth_path(path_vertices)
        else:
            final_path = path_vertices

        # 7. 可视化
        if visualize:
            self.visualizer.visualize_with_open3d(self.mesh, start_vertex, end_vertex, path_vertices, final_path,
                                                  collision_points)

        return final_path

    def set_algorithm(self, algorithm):
        """设置路径规划算法"""
        self.algorithm = algorithm
        return self


# 主程序入口
if __name__ == "__main__":
    # 替换为你的OBJ文件路径和两点坐标
    model_path = r"C:\Users\mi\Downloads\Three-Dimension-3D\models\pc\0\terra_obj\Block\Block.obj"
    start_point = np.array([78.56310916, 29.99410818, -162.10114156])  # 替换为实际坐标
    end_point = np.array([83.32050134, 42.84368528, -164.97868412])  # 替换为实际坐标

    # 创建路径规划器实例，默认使用A*算法
    planner = PathPlanner()
    # 加载网格
    planner.load_mesh(model_path)

    # 使用A*算法规划路径
    print("\n===== 使用A*算法 =====")
    path_astar = planner.plan_path(
        start_point,
        end_point,
        collision_safety_margin=0.5,
        visualize=True
    )

    if path_astar is not None:
        print(f"A*算法规划的路径包含 {len(path_astar)} 个点")
        # 可以保存路径数据
        np.savetxt('path_astar.csv', path_astar, delimiter=',', header='x,y,z')
        print("A*路径已保存到 path_astar.csv")
