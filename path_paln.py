import open3d as o3d
import numpy as np
import random
from scipy.spatial import KDTree

# 读取3D网格模型
mesh = o3d.io.read_triangle_mesh("model.obj")
mesh.compute_vertex_normals()

# 采样点云
point_cloud = mesh.sample_points_uniformly(number_of_points=5000)
pcd_points = np.asarray(point_cloud.points)

# 构建KDTree用于最近邻搜索
kdtree = KDTree(pcd_points)

# 定义起点、途径点、终点 (请替换为实际坐标)
A = np.array([1.0, 1.0, 1.0])  # 起点
B = np.array([2.0, 2.5, 1.5])  # 途径点
C = np.array([3.0, 3.0, 2.0])  # 终点


# 检查点是否在障碍物内（如果最近邻点太近，则视为碰撞）
def is_colliding(point, threshold=0.05):
    _, nearest_dist = kdtree.query(point)
    return nearest_dist < threshold


# RRT 采样函数
def sample_free_space():
    while True:
        sample = np.array([
            random.uniform(np.min(pcd_points[:, 0]), np.max(pcd_points[:, 0])),
            random.uniform(np.min(pcd_points[:, 1]), np.max(pcd_points[:, 1])),
            random.uniform(np.min(pcd_points[:, 2]), np.max(pcd_points[:, 2])),
        ])
        if not is_colliding(sample):
            return sample


# RRT 路径规划
def rrt(start, goal, max_iter=1000, step_size=0.1):
    tree = {tuple(start): None}  # 记录父子关系
    nodes = [start]

    for _ in range(max_iter):
        rand_point = sample_free_space()
        nearest_node = min(nodes, key=lambda node: np.linalg.norm(node - rand_point))
        direction = rand_point - nearest_node
        new_node = nearest_node + (direction / np.linalg.norm(direction)) * step_size

        if not is_colliding(new_node):
            nodes.append(new_node)
            tree[tuple(new_node)] = tuple(nearest_node)

            # 终点检查
            if np.linalg.norm(new_node - goal) < step_size:
                tree[tuple(goal)] = tuple(new_node)
                return reconstruct_path(tree, start, goal)

    return None  # 失败


# 路径重建
def reconstruct_path(tree, start, goal):
    path = [goal]
    current = tuple(goal)
    while current != tuple(start):
        current = tree[current]
        path.append(np.array(current))
    return path[::-1]  # 反转路径


# 计算 A -> B -> C 路径
path_A_B = rrt(A, B)
path_B_C = rrt(B, C)

if path_A_B and path_B_C:
    path = path_A_B + path_B_C[1:]  # 合并路径
    print("路径规划成功！")
else:
    print("路径规划失败！")
    exit()

# 绘制路径
lines = [[i, i + 1] for i in range(len(path) - 1)]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(path)
line_set.lines = o3d.utility.Vector2iVector(lines)

# 可视化
o3d.visualization.draw_geometries([mesh, line_set])
