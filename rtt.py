import numpy as np
import trimesh
import open3d as o3d
from rtree import index
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time


class PathPlanner:
    def __init__(self, obj_file_path, waypoints):
        print("Loading scene file...")
        start_time = time.time()
        self.mesh = trimesh.load_mesh(obj_file_path)
        print(f"Mesh loaded in {time.time() - start_time:.2f} seconds")

        self.waypoints = np.array(waypoints)
        print(f"Number of waypoints: {len(waypoints)}")

        print("Building spatial index...")
        start_time = time.time()
        self.spatial_index = self._build_spatial_index()
        print(f"Spatial index built in {time.time() - start_time:.2f} seconds")

        # 检查所有路径点的有效性
        self.validate_waypoints()

    def _build_spatial_index(self):
        p = index.Property()
        p.dimension = 3
        idx = index.Index(properties=p)

        for i, face in enumerate(self.mesh.faces):
            vertices = self.mesh.vertices[face]
            min_bounds = vertices.min(axis=0)
            max_bounds = vertices.max(axis=0)
            bounds = (
                float(min_bounds[0]), float(min_bounds[1]), float(min_bounds[2]),
                float(max_bounds[0]), float(max_bounds[1]), float(max_bounds[2])
            )
            idx.insert(i, bounds)
        return idx

    def validate_waypoints(self):
        """验证所有路径点的有效性"""
        print("\nValidating waypoints...")
        for i, point in enumerate(self.waypoints):
            collision = self.check_collision(point)
            distance = self.get_nearest_mesh_distance(point)
            print(f"Waypoint {i}: position={point}, collision={collision}, distance_to_mesh={distance:.3f}")

    def get_nearest_mesh_distance(self, point):
        """获取点到网格的最近距离"""
        query_point = np.array([point])
        closest_points, distances, _ = self.mesh.nearest.on_surface(query_point)
        return float(distances[0])

    def check_collision(self, point, radius=0.1):  # 减小碰撞检测半径
        """简化的碰撞检测"""
        # 获取点到网格的最近距离
        distance = self.get_nearest_mesh_distance(point)
        return distance < radius

    def _check_path_collision(self, point1, point2, checks=10):
        """检查路径碰撞"""
        vec = point2 - point1
        distance = np.linalg.norm(vec)
        direction = vec / distance
        step = distance / checks

        for i in range(checks + 1):
            check_point = point1 + direction * (i * step)
            if self.check_collision(check_point):
                return True
        return False

    def _plan_segment(self, start, end, max_iterations=2000, step_size=0.5):
        """RRT路径规划"""
        tree = {0: start}
        parents = {0: None}
        current_idx = 0

        pbar = tqdm(total=max_iterations, desc="Planning segment")

        def get_nearest_node(point):
            distances = [np.linalg.norm(point - node) for node in tree.values()]
            return min(range(len(distances)), key=distances.__getitem__)

        for iteration in range(max_iterations):
            # 采样策略
            if np.random.random() < 0.5:
                # 在目标方向采样
                random_point = end
            else:
                # 在空间中随机采样
                bounds_min = np.minimum(start, end) - 10.0
                bounds_max = np.maximum(start, end) + 10.0
                random_point = np.random.uniform(bounds_min, bounds_max)

            # 找到最近的节点
            nearest_idx = get_nearest_node(random_point)
            nearest_node = tree[nearest_idx]

            # 计算新节点
            direction = random_point - nearest_node
            distance = np.linalg.norm(direction)
            if distance > step_size:
                direction = direction / distance * step_size

            new_node = nearest_node + direction

            # 检查新节点和路径是否有效
            if not self.check_collision(new_node) and not self._check_path_collision(nearest_node, new_node):
                current_idx += 1
                tree[current_idx] = new_node
                parents[current_idx] = nearest_idx

                # 检查是否可以连接到目标点
                if np.linalg.norm(new_node - end) < step_size:
                    if not self._check_path_collision(new_node, end):
                        current_idx += 1
                        tree[current_idx] = end
                        parents[current_idx] = current_idx - 1
                        pbar.close()

                        # 重建路径
                        path = [end]
                        parent_idx = parents[current_idx]
                        while parent_idx is not None:
                            path.append(tree[parent_idx])
                            parent_idx = parents[parent_idx]
                        return path[::-1]

            pbar.update(1)

        pbar.close()
        return None

    def smooth_path(self, path, iterations=50):
        """平滑路径"""
        smoothed_path = np.array(path)

        for _ in range(iterations):
            for i in range(1, len(smoothed_path) - 1):
                current = smoothed_path[i]
                prev = smoothed_path[i - 1]
                next_point = smoothed_path[i + 1]

                new_pos = (prev + next_point) / 2

                if not self.check_collision(new_pos) and not self._check_path_collision(prev, new_pos) and not self._check_path_collision(new_pos, next_point):
                    smoothed_path[i] = new_pos

        return smoothed_path

    def plan_path(self):
        """路径规划主函数"""
        print("Planning path...")
        start_time = time.time()
        complete_path = []

        for i in range(len(self.waypoints) - 1):
            print(f"\nPlanning segment {i + 1}/{len(self.waypoints) - 1}")
            start = self.waypoints[i]
            end = self.waypoints[i + 1]

            # 尝试多次规划
            max_attempts = 3
            segment = None
            for attempt in range(max_attempts):
                print(f"Attempt {attempt + 1}/{max_attempts}")
                segment = self._plan_segment(start, end)
                if segment is not None:
                    break

            if segment is None:
                print(f"Failed to plan path between points {i} and {i + 1}")
                return None

            if complete_path:
                complete_path.extend(segment[1:])
            else:
                complete_path.extend(segment)

        if complete_path:
            print("\nSmoothing path...")
            smoothed_path = self.smooth_path(complete_path)
            print(f"\nPath planning completed in {time.time() - start_time:.2f} seconds")
            return smoothed_path

        return None

    def visualize(self, path=None):
        """可视化路径和网格"""
        print("Preparing visualization...")

        # 创建网格
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(self.mesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(self.mesh.faces)
        mesh_o3d.compute_vertex_normals()
        mesh_o3d.paint_uniform_color([0.8, 0.8, 0.8])  # 浅灰色网格

        geometries = [mesh_o3d]

        # 创建路径点
        waypoints_pcd = o3d.geometry.PointCloud()
        waypoints_pcd.points = o3d.utility.Vector3dVector(self.waypoints)
        waypoints_pcd.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in self.waypoints])  # 绿色路径点
        geometries.append(waypoints_pcd)

        # 如果有路径，添加路径线条和密集的可视化点
        if path is not None:
            # 创建沿路径的更密集点集来显示曲线
            dense_path = []
            for i in range(len(path) - 1):
                segment_points = np.linspace(path[i], path[i + 1], 20)  # 每段分为20个点
                dense_path.extend(segment_points)

            # 创建路径点云，显示为小球
            path_spheres = []
            for i, point in enumerate(path):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
                sphere.translate(point)
                if i == 0 or i == len(path) - 1:
                    sphere.paint_uniform_color([1, 0, 0])  # 红色起点和终点
                else:
                    sphere.paint_uniform_color([0, 0, 1])  # 蓝色中间点
                path_spheres.append(sphere)

            geometries.extend(path_spheres)

            # 创建每个点之间的线段
            lines = [[i, i + 1] for i in range(len(path) - 1)]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(path)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[1, 0.7, 0] for _ in lines])  # 橙色路径
            geometries.append(line_set)

            # 也可以添加一个更密集的点云来显示曲线
            curve_pcd = o3d.geometry.PointCloud()
            curve_pcd.points = o3d.utility.Vector3dVector(dense_path)
            curve_pcd.colors = o3d.utility.Vector3dVector([[1, 0.7, 0] for _ in dense_path])  # 橙色点
            curve_pcd.estimate_normals()
            geometries.append(curve_pcd)

        print("Showing visualization...")
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # 添加所有几何体
        for geom in geometries:
            vis.add_geometry(geom)

        # 设置相机位置和视角
        view_control = vis.get_view_control()
        view_control.set_zoom(0.8)

        # 运行可视化
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    obj_file = r"C:\Users\mi\Downloads\Three-Dimension-3D\models\pc\0\terra_obj\Block\Block.obj"
    waypoints = [
        [78.56310916, 29.99410818, -162.10114156],
        [83.32050134, 42.84368528, -164.97868412],
        [94.56436536, 52.84385806, -162.24338607],
        [77.65503504, 54.38683771, -166.85624655]
    ]

    print("Initializing path planner...")
    planner = PathPlanner(obj_file, waypoints)

    # 先可视化原始点和网格
    print("\nVisualizing initial configuration...")
    planner.visualize()

    # 进行路径规划
    path = planner.plan_path()
    if path is not None:
        print("\nVisualizing planned path...")
        planner.visualize(path)
    else:
        print("Failed to find valid path")