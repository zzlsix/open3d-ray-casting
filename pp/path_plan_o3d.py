import open3d as o3d
import numpy as np
import random
import time

model_path = r"C:\Users\mi\Downloads\Three-Dimension-3D\models\pc\0\terra_obj\Block\Block.obj"


def main():
    # 1. 导入OBJ格式模型文件
    print("正在加载3D模型...")
    mesh = o3d.io.read_triangle_mesh(model_path)

    # 确保模型有法线信息，用于碰撞检测
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # 可视化模型
    print("模型信息:")
    print(f"- 顶点数量: {len(mesh.vertices)}")
    print(f"- 三角面数量: {len(mesh.triangles)}")

    # 2. 定义起点和终点(假设A和B是模型表面上的两个点)
    # 在实际应用中，这些点应该是模型表面上的有效点
    point_A = np.array([78.56310916, 29.99410818, -162.10114156])  # 这里用示例坐标，请替换为实际坐标
    point_B = np.array([83.32050134, 42.84368528, -164.97868412])  # 这里用示例坐标，请替换为实际坐标

    # 确保A和B在模型表面上
    point_A = project_point_to_mesh(point_A, mesh)
    point_B = project_point_to_mesh(point_B, mesh)

    print(f"起点A: {point_A}")
    print(f"终点B: {point_B}")

    # 3. 构建用于碰撞检测的体素网格
    print("正在构建碰撞检测模型...")

    # 基于模型尺寸计算
    vertices = np.asarray(mesh.vertices)
    min_bound = np.min(vertices, axis=0)
    max_bound = np.max(vertices, axis=0)

    # 计算模型直径
    model_diameter = np.linalg.norm(max_bound - min_bound)
    print(f"模型直径: {model_diameter:.4f}")

    # 体素大小，根据模型尺寸调整
    voxel_size = model_diameter / 50
    print(f"体素尺寸: {voxel_size:.4f}")

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)

    # 4. 调用RRT算法进行路径规划
    print("开始路径规划...")
    start_time = time.time()
    path = rrt_path_planning(point_A, point_B, mesh, voxel_grid)
    end_time = time.time()

    if path is None:
        print("未能找到有效路径!")
        return

    print(f"路径规划完成，耗时: {end_time - start_time:.2f}秒")
    print(f"路径点数量: {len(path)}")

    # 5. 可视化结果
    visualize_path(mesh, path, point_A, point_B)


def project_point_to_mesh(point, mesh):
    """将点投影到网格表面上"""
    # 构建KD树用于最近点搜索
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # 寻找最近的顶点
    k, idx, _ = kdtree.search_knn_vector_3d(point, 1)

    # 返回网格上的点
    return np.asarray(mesh.vertices)[idx[0]]


def check_collision(point, mesh, voxel_grid, safety_margin=0.01):
    """检查点是否发生碰撞"""
    # 使用体素网格进行快速碰撞检测
    # 如果点未投影到表面，则认为是碰撞
    # 使用KD树检查点是否接近表面
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # 找到最近的顶点
    k, idx, squared_distance = kdtree.search_knn_vector_3d(point, 1)

    # 如果点太远离表面，则认为是碰撞(不在表面上)
    # 或者点在网格内部，也认为是碰撞
    distance = np.sqrt(squared_distance[0])
    if distance > safety_margin:
        return True

    # 获取最近顶点的法线
    vertex_idx = idx[0]
    vertex_normal = np.asarray(mesh.vertex_normals)[vertex_idx]

    # 获取从顶点到点的方向
    direction = point - np.asarray(mesh.vertices)[vertex_idx]

    # 如果方向与法线方向相反，则点可能在网格内部
    if np.dot(direction, vertex_normal) < 0:
        return True

    return False


def check_segment_collision(p1, p2, mesh, voxel_grid, steps=10):
    """检查线段是否发生碰撞"""
    for i in range(steps + 1):
        t = i / steps
        point = p1 * (1 - t) + p2 * t
        if check_collision(point, mesh, voxel_grid):
            return True
    return False


def rrt_path_planning(start, goal, mesh, voxel_grid, max_iterations=1000, step_size=7):
    """使用RRT算法在网格表面上规划路径"""
    # 初始化RRT树
    vertices = [start]
    edges = []

    # 获取模型边界，用于采样
    min_bound = np.min(np.asarray(mesh.vertices), axis=0)
    max_bound = np.max(np.asarray(mesh.vertices), axis=0)

    for i in range(max_iterations):
        # 显示进度
        if i % 100 == 0:
            print(f"迭代次数: {i}/{max_iterations}")

        # 随机采样一个点
        if random.random() < 0.1:  # 10%的概率直接采样终点
            sample = goal
        else:
            # 随机采样空间中的点
            sample = np.random.uniform(min_bound, max_bound)
            # 将点投影到网格表面
            sample = project_point_to_mesh(sample, mesh)

        # 寻找最近的顶点
        nearest_idx = find_nearest_vertex(sample, vertices)
        nearest_vertex = vertices[nearest_idx]

        # 朝采样点方向扩展
        direction = sample - nearest_vertex
        dist = np.linalg.norm(direction)
        if dist > step_size:
            direction = direction / dist * step_size

        new_vertex = nearest_vertex + direction

        # 将新点投影到表面
        new_vertex = project_point_to_mesh(new_vertex, mesh)

        # 检查新节点是否发生碰撞
        if check_collision(new_vertex, mesh, voxel_grid):
            continue

        # 检查路径是否发生碰撞
        if check_segment_collision(nearest_vertex, new_vertex, mesh, voxel_grid):
            continue

        # 添加新节点和边
        vertices.append(new_vertex)
        edges.append((nearest_idx, len(vertices) - 1))

        # 检查是否可以连接到终点
        if np.linalg.norm(new_vertex - goal) < step_size * 2:
            if not check_segment_collision(new_vertex, goal, mesh, voxel_grid):
                vertices.append(goal)
                edges.append((len(vertices) - 2, len(vertices) - 1))

                # 找到路径
                return extract_path(vertices, edges, len(vertices) - 1)

    print("达到最大迭代次数，未找到路径。")
    return None


def find_nearest_vertex(point, vertices):
    """找到最近的顶点索引"""
    distances = [np.linalg.norm(point - v) for v in vertices]
    return np.argmin(distances)


def extract_path(vertices, edges, goal_idx):
    """从RRT树中提取路径"""
    path = [vertices[goal_idx]]
    current_idx = goal_idx

    # 构建从子节点到父节点的映射
    parent_map = {}
    for edge in edges:
        parent, child = edge
        parent_map[child] = parent

    # 回溯路径
    while current_idx in parent_map:
        current_idx = parent_map[current_idx]
        path.append(vertices[current_idx])

    # 反转路径，使其从起点到终点
    return path[::-1]


def visualize_path(mesh, path, start, end):
    """可视化路径和模型"""
    # 创建可视化对象
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加网格
    vis.add_geometry(mesh)

    # 创建路径线段
    path_points = o3d.utility.Vector3dVector(path)
    line_indices = [[i, i + 1] for i in range(len(path) - 1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = path_points
    line_set.lines = o3d.utility.Vector2iVector(line_indices)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(line_indices))])  # 红色路径
    vis.add_geometry(line_set)

    # 创建起点和终点球体
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    start_sphere.translate(start)
    start_sphere.paint_uniform_color([0, 1, 0])  # 绿色起点
    vis.add_geometry(start_sphere)

    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    end_sphere.translate(end)
    end_sphere.paint_uniform_color([0, 0, 1])  # 蓝色终点
    vis.add_geometry(end_sphere)

    # 设置视图
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 5.0

    # 运行可视化
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()