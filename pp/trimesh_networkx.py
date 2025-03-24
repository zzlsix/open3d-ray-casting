import trimesh
import networkx as nx
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_path_between_points(mesh_path, start_point, end_point,
                               collision_safety_margin=0.01,
                               visualize=True,
                               connect_components=True,
                               max_connection_distance=10.0):
    """
    在3D模型上找到两点之间的路径，并进行碰撞检测

    参数:
    mesh_path - .obj文件的路径
    start_point - 起始点坐标 [x, y, z]
    end_point - 终点坐标 [x, y, z]
    collision_safety_margin - 碰撞检测的安全距离
    visualize - 是否可视化路径
    connect_components - 是否尝试连接不连通的组件
    max_connection_distance - 最大连接距离
    """
    # 1. 加载3D模型
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
            mesh = trimesh.util.concatenate(geometries)
        else:
            mesh = geometries[0]
    else:
        mesh = loaded_obj

    # 2. 检查模型属性
    if hasattr(mesh, 'is_watertight') and not mesh.is_watertight:
        print("警告: 模型不是水密的，可能会影响路径规划质量")

    # 3. 寻找最接近起点和终点的顶点
    print("查找模型上的起点和终点...")
    vertices = mesh.vertices
    kdtree = KDTree(vertices)

    # 查找多个最近点而不只是一个，以防最近点不在主要连通分量上
    k_nearest = 5  # 查找5个最近点
    start_dists, start_indices = kdtree.query(start_point, k=k_nearest)
    end_dists, end_indices = kdtree.query(end_point, k=k_nearest)

    # 初始设为第一个最近点
    start_idx = start_indices[0]
    end_idx = end_indices[0]

    start_vertex = vertices[start_idx]
    end_vertex = vertices[end_idx]

    print(f"最接近的起点: {start_vertex}, 索引: {start_idx}")
    print(f"最接近的终点: {end_vertex}, 索引: {end_idx}")

    # 4. 使用网格拓扑创建图
    print("构建网络图...")
    graph = nx.Graph()

    # 添加所有顶点
    for i, vertex in enumerate(vertices):
        graph.add_node(i, position=vertex)

    # 添加所有边，使用网格的边连接信息
    if hasattr(mesh, 'edges_unique'):
        edges = mesh.edges_unique
    else:
        # 如果没有edges_unique属性，尝试从面构建边
        edges = set()
        for face in mesh.faces:
            for i in range(len(face)):
                edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
                edges.add(edge)
        edges = list(edges)

    for edge in edges:
        v1, v2 = edge
        # 计算边的长度作为权重
        weight = np.linalg.norm(vertices[v1] - vertices[v2])
        graph.add_edge(v1, v2, weight=weight)

    # 5. 分析图的连通性
    print("分析图的连通性...")
    connected_components = list(nx.connected_components(graph))
    print(f"发现 {len(connected_components)} 个连通分量")

    # 找出起点和终点所在的连通分量
    start_component = None
    end_component = None

    for i, component in enumerate(connected_components):
        if start_idx in component:
            start_component = i
        if end_idx in component:
            end_component = i

    print(f"起点在连通分量 {start_component}，终点在连通分量 {end_component}")

    # 6. 处理连通性问题
    if start_component != end_component and start_component is not None and end_component is not None:
        print("起点和终点不在同一连通分量中")

        if connect_components:
            print("尝试连接不同的连通分量...")

            # 策略1: 尝试不同的起点和终点候选
            found_connection = False
            for s_idx in start_indices:
                for e_idx in end_indices:
                    # 找出这些点所在的连通分量
                    s_comp = None
                    e_comp = None
                    for i, component in enumerate(connected_components):
                        if s_idx in component:
                            s_comp = i
                        if e_idx in component:
                            e_comp = i

                    if s_comp == e_comp and s_comp is not None:
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
                print("添加连通分量之间的连接...")

                # 为每个组件计算一个中心点
                component_centers = []
                for component in connected_components:
                    comp_vertices = [vertices[i] for i in component]
                    center = np.mean(comp_vertices, axis=0)
                    component_centers.append((center, list(component)[0]))  # 中心和一个代表点

                # 连接所有组件到其最近的N个组件
                n_connections = min(3, len(connected_components))
                for i, (center_i, rep_i) in enumerate(component_centers):
                    # 计算到其他中心的距离
                    distances = []
                    for j, (center_j, rep_j) in enumerate(component_centers):
                        if i != j:
                            dist = np.linalg.norm(center_i - center_j)
                            distances.append((dist, j, rep_j))

                    # 连接到最近的N个组件
                    distances.sort()
                    for d, j, rep_j in distances[:n_connections]:
                        if d <= max_connection_distance:
                            print(f"连接组件 {i} 到 {j}, 距离: {d:.2f}")
                            graph.add_edge(rep_i, rep_j, weight=d)

                # 检查起点和终点现在是否连通
                if nx.has_path(graph, start_idx, end_idx):
                    print("成功连接起点和终点所在的连通分量")
                else:
                    # 最后的尝试：直接连接起点和终点
                    direct_distance = np.linalg.norm(start_vertex - end_vertex)
                    if direct_distance <= max_connection_distance:
                        print(f"直接连接起点和终点，距离: {direct_distance:.2f}")
                        graph.add_edge(start_idx, end_idx, weight=direct_distance)

    # 7. 使用A*算法找到路径
    print("使用A*算法规划路径...")
    try:
        # 定义启发式函数
        def heuristic(a, b):
            pos_a = graph.nodes[a]['position']
            pos_b = graph.nodes[b]['position']
            return np.linalg.norm(pos_a - pos_b)

        if not nx.has_path(graph, start_idx, end_idx):
            raise nx.NetworkXNoPath("起点和终点之间没有路径")

        path_indices = nx.astar_path(graph, start_idx, end_idx, heuristic)
        print(f"找到路径，长度: {len(path_indices)}个点")

        # 提取路径的顶点坐标
        path_vertices = [vertices[idx] for idx in path_indices]

    except nx.NetworkXNoPath:
        print("无法找到路径，可能需要调整连接参数或选择不同的起点和终点")

        if visualize:
            # 尽管没有路径，但仍然可视化网格和两点
            visualize_no_path(mesh, start_vertex, end_vertex, connected_components, vertices)

        return None

    # 8. 碰撞检测
    print("进行碰撞检测...")
    collision_points = []

    for i in range(len(path_indices) - 1):
        v1 = vertices[path_indices[i]]
        v2 = vertices[path_indices[i + 1]]

        # 使用更精确的碰撞检测
        if ray_collision_check(mesh, v1, v2, collision_safety_margin):
            print(f"检测到碰撞在路径段 {i} 到 {i + 1}")
            collision_points.append((i, v1, v2))

    if collision_points:
        print(f"发现 {len(collision_points)} 个碰撞点")

        # 处理碰撞点 - 简单方法是从路径中移除碰撞段
        if len(collision_points) < len(path_indices) - 1:  # 确保不会移除所有路径
            new_path_indices = []
            skip_mode = False

            for i in range(len(path_indices)):
                if i < len(path_indices) - 1:
                    # 检查当前边是否是碰撞边
                    is_collision_edge = any(cp[0] == i for cp in collision_points)

                    if is_collision_edge:
                        skip_mode = True
                    elif skip_mode:
                        skip_mode = False
                        new_path_indices.append(path_indices[i])
                    else:
                        new_path_indices.append(path_indices[i])
                else:
                    # 总是保留最后一个点
                    new_path_indices.append(path_indices[i])

            path_indices = new_path_indices
            print(f"移除碰撞段后的路径长度: {len(path_indices)}")

            # 重新提取路径顶点
            path_vertices = [vertices[idx] for idx in path_indices]
        else:
            print("警告: 所有路径段都存在碰撞，无法简单修复")
    else:
        print("路径无碰撞")

    # 9. 平滑路径
    if len(path_indices) > 2:
        print("平滑路径...")
        # 这里可以实现更复杂的路径平滑算法
        final_path = smooth_path(path_vertices, mesh, collision_safety_margin)
    else:
        final_path = path_vertices

    # 10. 可视化
    if visualize:
        visualize_with_open3d(mesh, start_vertex, end_vertex, path_vertices, final_path, collision_points)

    return final_path


def ray_collision_check(mesh, start_point, end_point, safety_margin=0.01):
    """更精确的射线碰撞检测"""
    direction = end_point - start_point
    distance = np.linalg.norm(direction)

    if distance < 1e-6:  # 防止距离过小导致的问题
        return False

    direction = direction / distance

    # 在起点略微偏移以避免自相交
    offset_start = start_point + direction * safety_margin

    # 设置射线长度略小于总距离，避免终点处的误检
    ray_length = distance - 2 * safety_margin

    if ray_length <= 0:
        return False  # 太短的线段视为无碰撞

    # 使用更多的中间采样点进行检测
    samples = min(10, max(2, int(distance / 0.5)))  # 根据距离确定采样数

    for i in range(samples):
        # 创建均匀分布的采样点
        t = i / (samples - 1)
        ray_origin = offset_start + t * direction * ray_length

        # 检查向下一个采样点的射线
        if i < samples - 1:
            next_origin = offset_start + (i + 1) / (samples - 1) * direction * ray_length
            sub_direction = next_origin - ray_origin
            sub_length = np.linalg.norm(sub_direction)

            if sub_length > 0:
                sub_direction = sub_direction / sub_length
                try:
                    hits = mesh.ray.intersects_any(
                        ray_origins=[ray_origin],
                        ray_directions=[sub_direction],
                        ray_distances=[sub_length]
                    )
                    if any(hits):
                        return True  # 检测到碰撞
                except (AttributeError, ValueError) as e:
                    print(f"射线检测异常: {e}")

    return False  # 未检测到碰撞

def smooth_path(path, mesh, safety_margin=0.01, iterations=2):
    """
    使用移动平均和直线可行性检查来平滑路径

    参数:
    path - 原始路径点列表
    mesh - 3D模型
    safety_margin - 碰撞检测安全距离
    iterations - 平滑迭代次数
    """
    if len(path) <= 2:
        return path

    smoothed = path.copy()

    for _ in range(iterations):
        # 移动平均平滑
        new_path = [smoothed[0]]  # 保持起点不变

        for i in range(1, len(smoothed) - 1):
            # 当前点和相邻点的加权平均
            current = np.array(smoothed[i])
            prev = np.array(smoothed[i - 1])
            next_pt = np.array(smoothed[i + 1])

            # 加权平均 (0.5 当前点, 0.25 前一点, 0.25 后一点)
            avg = current * 0.5 + prev * 0.25 + next_pt * 0.25

            # 检查平滑点是否会导致碰撞
            collision = False
            for neighbor in [prev, next_pt]:
                ray_origin = avg
                ray_direction = neighbor - avg
                ray_length = np.linalg.norm(ray_direction)

                if ray_length > 0:
                    ray_direction = ray_direction / ray_length
                    try:
                        hits = mesh.ray.intersects_any(
                            ray_origins=[ray_origin + ray_direction * safety_margin],
                            ray_directions=[ray_direction]
                        )
                        collision = collision or any(hits)
                    except (AttributeError, ValueError):
                        # 如果无法进行射线检测，保守地假设无碰撞
                        pass

            # 如果平滑点会导致碰撞，保留原始点
            if collision:
                new_path.append(smoothed[i])
            else:
                new_path.append(avg)

        new_path.append(smoothed[-1])  # 保持终点不变
        smoothed = new_path

        # 跳过冗余点
        i = 0
        while i < len(smoothed) - 2:
            # 如果三个连续点几乎共线，可以去掉中间点
            v1 = smoothed[i + 1] - smoothed[i]
            v2 = smoothed[i + 2] - smoothed[i + 1]
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)

            if len1 > 0 and len2 > 0:
                cosine = np.dot(v1, v2) / (len1 * len2)
                if cosine > 0.99:  # 几乎共线 (cos(~8°) ≈ 0.99)
                    # 检查直接连接是否可行
                    direct = smoothed[i + 2] - smoothed[i]
                    direct_len = np.linalg.norm(direct)
                    if direct_len > 0:
                        direct_norm = direct / direct_len
                        try:
                            hits = mesh.ray.intersects_any(
                                ray_origins=[smoothed[i] + direct_norm * safety_margin],
                                ray_directions=[direct_norm]
                            )
                            if not any(hits):
                                # 可以安全地移除中间点
                                smoothed.pop(i + 1)
                                continue
                        except (AttributeError, ValueError):
                            pass
            i += 1

    return smoothed

def visualize_no_path(mesh, start, end, connected_components, vertices):
    """可视化网格、起点、终点和连通分量"""
    print("可视化网格和连通分量...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制起点和终点
    ax.scatter([start[0]], [start[1]], [start[2]], color='blue', s=100, label='起点')
    ax.scatter([end[0]], [end[1]], [end[2]], color='green', s=100, label='终点')

    # 绘制不同连通分量的部分顶点
    colors = plt.cm.rainbow(np.linspace(0, 1, len(connected_components)))
    for i, component in enumerate(connected_components):
        # 从每个分量中取样一些点来可视化
        sample_size = min(100, len(component))
        sample = np.random.choice(list(component), sample_size, replace=False)
        component_vertices = vertices[sample]
        ax.scatter(
            component_vertices[:, 0],
            component_vertices[:, 1],
            component_vertices[:, 2],
            color=colors[i],
            alpha=0.5,
            s=10,
            label=f'组件 {i} ({len(component)} 个点)'
        )

    ax.set_title('3D模型 - 无法找到路径')
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_with_open3d(mesh, start, end, original_path, smoothed_path=None, collision_points=None):
    """使用Open3D可视化3D模型和路径"""
    try:
        import open3d as o3d
    except ImportError:
        print("需要安装Open3D: pip install open3d")
        return

    # 将Trimesh转换为Open3D网格
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()

    # 创建原始路径的线集合
    path_array = np.array(original_path)
    path_lines = o3d.geometry.LineSet()
    path_lines.points = o3d.utility.Vector3dVector(path_array)
    path_lines.lines = o3d.utility.Vector2iVector(
        [[i, i + 1] for i in range(len(path_array) - 1)]
    )
    path_lines.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(path_array) - 1)])

    # 创建平滑路径的线集合
    smooth_lines = None
    if smoothed_path is not None and id(smoothed_path) != id(original_path):
        smooth_array = np.array(smoothed_path)
        smooth_lines = o3d.geometry.LineSet()
        smooth_lines.points = o3d.utility.Vector3dVector(smooth_array)
        smooth_lines.lines = o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(smooth_array) - 1)]
        )
        smooth_lines.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(smooth_array) - 1)])

    # 创建起点和终点球体
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    start_sphere.translate(start)
    start_sphere.paint_uniform_color([0, 0, 1])  # 蓝色

    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    end_sphere.translate(end)
    end_sphere.paint_uniform_color([0, 1, 0])  # 绿色

    # 创建路径点的小球
    path_spheres = []
    for i in range(0, len(path_array), max(1, len(path_array) // 20)):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        sphere.translate(path_array[i])
        sphere.paint_uniform_color([1, 1, 0])
        path_spheres.append(sphere)

    # 显示所有几何体
    geometries = [o3d_mesh, path_lines, start_sphere, end_sphere] + path_spheres
    if smooth_lines is not None:
        geometries.append(smooth_lines)

    # 如果有碰撞点，显示它们
    if collision_points and len(collision_points) > 0:
        for i, start_pt, end_pt in collision_points:
            # 创建红色球体标记碰撞点
            collision_sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.8)
            collision_sphere1.translate(start_pt)
            collision_sphere1.paint_uniform_color([1, 0, 0])  # 红色

            collision_sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.8)
            collision_sphere2.translate(end_pt)
            collision_sphere2.paint_uniform_color([1, 0, 0])  # 红色

            # 添加到几何体列表
            geometries.extend([collision_sphere1, collision_sphere2])

    # 设置渲染选项并显示
    o3d.visualization.draw_geometries(
        geometries,
        window_name="3D Model Path Planning",
        width=1024,
        height=768,
        point_show_normal=False,
        mesh_show_wireframe=True,
        mesh_show_back_face=False,
    )

# 使用示例
if __name__ == "__main__":

    # 替换为你的OBJ文件路径和两点坐标
    model_path = r"C:\Users\mi\Downloads\Three-Dimension-3D\models\pc\0\terra_obj\Block\Block.obj"
    start_point = np.array([78.56310916, 29.99410818, -162.10114156])  # 替换为实际坐标
    end_point = np.array([83.32050134, 42.84368528, -164.97868412])  # 替换为实际坐标

    # 增加安全边距以更容易检测碰撞
    path = create_path_between_points(
        model_path,
        start_point,
        end_point,
        collision_safety_margin=0.5,  # 增加安全边距
        visualize=True
    )

    if path is not None:
        print(f"最终路径包含 {len(path)} 个点")
        print(f"最终路径包含 {path} 个点")
        # 可以保存路径数据
        np.savetxt('path.csv', path, delimiter=',', header='x,y,z')
        print("路径已保存到 path.csv")