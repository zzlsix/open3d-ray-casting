import heapq
import time
from collections import defaultdict

import numpy as np
import trimesh

from path_planning_old.visualizer.PathVisualizer import PathVisualizer


def load_and_voxelize_obj(obj_file, voxel_size=0.1):
    """加载OBJ文件并将其体素化"""
    print(f"加载模型: {obj_file}")

    # 加载模型，可能是Scene或Mesh
    model = trimesh.load(obj_file)

    # 处理Scene对象 - 提取所有网格并合并
    if isinstance(model, trimesh.Scene):
        print("检测到Scene对象，合并所有网格...")
        # 获取场景中的所有几何体
        geometries = list(model.geometry.values())
        if not geometries:
            raise ValueError("无法从模型中提取几何体")

        # 如果只有一个几何体，直接使用它
        if len(geometries) == 1:
            mesh = geometries[0]
        else:
            # 否则合并所有几何体
            mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = model

    print(f"模型处理完成，开始体素化...")

    # 计算模型边界
    bounds = mesh.bounds
    print(f"模型边界: {bounds}")
    print(f"模型尺寸: {bounds[1] - bounds[0]}")

    # 体素化网格
    voxel_grid = mesh.voxelized(pitch=voxel_size)
    occupied_voxels = set(map(tuple, voxel_grid.points))
    print(f"生成了 {len(occupied_voxels)} 个占用体素")

    # 返回网格、体素和边界
    return mesh, occupied_voxels, bounds


def find_nearest_surface_point(mesh, point, search_radius=10.0):
    """找到距离给定点最近的表面点，但在表面外部"""
    # 计算点到网格的最近点
    closest_point, distance, triangle_id = trimesh.proximity.closest_point(mesh, [point])
    closest_point = closest_point[0]
    distance = distance[0]

    print(f"到表面的距离: {distance}")

    # 如果点在表面上（距离很小）
    if distance < 1e-5:
        # 获取该点所在三角形的法向量
        normal = mesh.face_normals[triangle_id[0]]

        # 沿法向量移动一段距离，以确保在表面外部
        offset_distance = 1.5  # 这个值可能需要调整
        safe_point = point + normal * offset_distance
        print(f"表面点已偏移: 从 {point} 到 {safe_point}")
        return safe_point

    # 如果点不在表面上但靠近内部
    if is_point_inside_mesh(mesh, point):
        # 沿着最近表面点方向移动
        direction = np.array(closest_point) - np.array(point)
        direction = direction / np.linalg.norm(direction)  # 归一化

        # 计算安全点，稍微超出表面
        offset_distance = distance + 1.0  # 额外偏移
        safe_point = point + direction * offset_distance
        print(f"内部点已移到外部: 从 {point} 到 {safe_point}")
        return safe_point

    # 如果点已经在外部，保持不变
    return point


def is_point_inside_mesh(mesh, point):
    """检查点是否在网格内部"""
    # 创建从点出发的射线
    ray_origins = np.array([point])
    ray_directions = np.array([[1.0, 0.0, 0.0]])  # 沿x轴正方向

    # 计算射线与网格的交点数
    intersections = mesh.ray.intersects_location(ray_origins, ray_directions)

    # 如果交点数为奇数，则点在内部
    return len(intersections[0]) % 2 == 1


def is_point_in_voxels(point, occupied_voxels, voxel_size):
    """检查点是否在占用体素中"""
    grid_point = tuple(np.round(np.array(point) / voxel_size).astype(int))
    world_point = tuple(np.array(grid_point) * voxel_size)
    return world_point in occupied_voxels


def heuristic(a, b):
    """计算两点间的欧几里得距离"""
    return np.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def astar_3d_surface(mesh, start, goal, occupied_voxels, voxel_size=0.1, bounds=None):
    """
    修改版A*算法，处理模型表面上的起点和终点
    """
    # 处理起点和终点，确保它们在表面外部
    safe_start = find_nearest_surface_point(mesh, start)
    safe_goal = find_nearest_surface_point(mesh, goal)

    print(f"处理后的起点: {safe_start}")
    print(f"处理后的终点: {safe_goal}")

    # 将连续坐标转换为离散网格坐标
    def to_grid(point):
        return tuple(np.round(np.array(point) / voxel_size).astype(int))

    def to_world(grid_point):
        return tuple(np.array(grid_point) * voxel_size)

    # 检查点是否在边界内
    def is_within_bounds(point):
        if bounds is None:
            return True
        # 增加一点余量，因为我们的点可能在边界外一点点
        margin = voxel_size * 2
        return all(bounds[0][i] - margin <= point[i] <= bounds[1][i] + margin for i in range(3))

    # 检查点是否与障碍物碰撞
    def is_collision(point):
        return is_point_in_voxels(point, occupied_voxels, voxel_size)

    print(f"开始A*搜索，从 {safe_start} 到 {safe_goal}")
    start_time = time.time()

    # 初始化
    start_grid = to_grid(safe_start)
    goal_grid = to_grid(safe_goal)

    # 再次检查起点和终点是否有效
    if is_collision(safe_start):
        print(f"警告：处理后的起点仍在障碍物内，尝试小幅调整...")
        for offset in [
            (voxel_size, 0, 0), (-voxel_size, 0, 0),
            (0, voxel_size, 0), (0, -voxel_size, 0),
            (0, 0, voxel_size), (0, 0, -voxel_size)
        ]:
            new_start = tuple(np.array(safe_start) + np.array(offset))
            if not is_collision(new_start):
                safe_start = new_start
                start_grid = to_grid(safe_start)
                print(f"起点已调整为: {safe_start}")
                break

    if is_collision(safe_goal):
        print(f"警告：处理后的终点仍在障碍物内，尝试小幅调整...")
        for offset in [
            (voxel_size, 0, 0), (-voxel_size, 0, 0),
            (0, voxel_size, 0), (0, -voxel_size, 0),
            (0, 0, voxel_size), (0, 0, -voxel_size)
        ]:
            new_goal = tuple(np.array(safe_goal) + np.array(offset))
            if not is_collision(new_goal):
                safe_goal = new_goal
                goal_grid = to_grid(safe_goal)
                print(f"终点已调整为: {safe_goal}")
                break

    # 如果仍然无法找到有效的起点或终点
    if is_collision(safe_start):
        print("错误：无法找到有效的起点")
        return None

    if is_collision(safe_goal):
        print("错误：无法找到有效的终点")
        return None

    open_set = []
    heapq.heappush(open_set, (0, start_grid))

    came_from = {}
    g_score = defaultdict(lambda: float('inf'))
    g_score[start_grid] = 0

    f_score = defaultdict(lambda: float('inf'))
    f_score[start_grid] = heuristic(start_grid, goal_grid)

    open_set_hash = {start_grid}

    visited_count = 0

    while open_set:
        # 获取f值最小的节点
        current_f, current = heapq.heappop(open_set)
        open_set_hash.remove(current)
        visited_count += 1

        # 定期显示进度
        if visited_count % 1000 == 0:
            print(f"已探索 {visited_count} 个节点...")

        # 达到目标
        if current == goal_grid:
            path = []
            while current in came_from:
                path.append(to_world(current))
                current = came_from[current]
            path.append(safe_start)
            path.reverse()

            # 在路径末尾添加真正的目标点
            path.append(goal)

            end_time = time.time()
            print(f"找到路径！用时: {end_time - start_time:.2f}秒")
            print(f"路径长度: {len(path)}个点")
            print(f"探索节点数: {visited_count}")
            return path

        # 检查所有邻居
        for neighbor_grid in [
            (current[0] + dx, current[1] + dy, current[2] + dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if not (dx == 0 and dy == 0 and dz == 0)
        ]:
            # 检查邻居点是否合法
            neighbor_world = to_world(neighbor_grid)

            if not is_within_bounds(neighbor_world) or is_collision(neighbor_world):
                continue

            # 计算到达邻居的代价
            # 对角线移动距离与直线移动距离不同
            dx, dy, dz = abs(current[0] - neighbor_grid[0]), abs(current[1] - neighbor_grid[1]), abs(
                current[2] - neighbor_grid[2])

            if dx + dy + dz == 1:  # 直线移动
                move_cost = voxel_size
            elif dx + dy + dz == 2:  # 面对角线
                move_cost = 1.414 * voxel_size  # √2
            else:  # 体对角线
                move_cost = 1.732 * voxel_size  # √3

            tentative_g = g_score[current] + move_cost

            # 如果找到更好的路径
            if tentative_g < g_score[neighbor_grid]:
                came_from[neighbor_grid] = current
                g_score[neighbor_grid] = tentative_g
                f_score[neighbor_grid] = tentative_g + heuristic(neighbor_grid, goal_grid)

                if neighbor_grid not in open_set_hash:
                    heapq.heappush(open_set, (f_score[neighbor_grid], neighbor_grid))
                    open_set_hash.add(neighbor_grid)

    # 如果没有找到路径
    end_time = time.time()
    print(f"没有找到路径。用时: {end_time - start_time:.2f}秒")
    print(f"探索节点数: {visited_count}")
    return None


def post_process_path(mesh, path, original_start, original_end):
    """将路径点投影到表面附近"""
    processed_path = []

    # 保留原始起点和终点
    processed_path.append(original_start)

    # 处理中间点 - 将它们投影到距离表面更近的位置
    for i in range(1, len(path) - 1):
        point = path[i]

        # 计算点到网格的最近点
        closest_point, distance, triangle_id = trimesh.proximity.closest_point(mesh, [point])
        closest_point = closest_point[0]

        # 计算表面法线
        normal = mesh.face_normals[triangle_id[0]]

        # 将点移动到表面附近但不在表面上
        # 可以调整这个偏移量使路径更接近表面
        offset_distance = 0.2  # 小偏移，让路径非常接近表面
        projected_point = closest_point + normal * offset_distance

        processed_path.append(projected_point)

    # 添加原始终点
    processed_path.append(original_end)

    return processed_path


def estimate_good_voxel_size(obj_file):
    """估计合适的体素大小"""
    model = trimesh.load(obj_file)

    # 处理Scene对象
    if isinstance(model, trimesh.Scene):
        geometries = list(model.geometry.values())
        if not geometries:
            raise ValueError("无法从模型中提取几何体")

        if len(geometries) == 1:
            mesh = geometries[0]
        else:
            mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = model

    # 计算模型边界和尺寸
    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]

    # 取模型最小尺寸的1%作为体素大小
    suggested_size = min(dimensions) * 0.01

    # 设置上下限
    min_size = 0.01  # 最小体素大小，单位与模型相同
    max_size = 1.0  # 最大体素大小

    voxel_size = max(min_size, min(max_size, suggested_size))

    print(f"模型尺寸: {dimensions}")
    print(f"建议的体素大小: {voxel_size}")

    return voxel_size


def main():
    # 配置参数
    obj_file = r"C:\Users\mi\Downloads\Three-Dimension-3D\models\pc\0\terra_obj\Block\Block.obj"  # 替换为你的OBJ文件路径

    # 估计合适的体素大小
    voxel_size = estimate_good_voxel_size(obj_file)

    # 加载并体素化OBJ模型
    mesh, occupied_voxels, bounds = load_and_voxelize_obj(obj_file, voxel_size)

    # 用户提供的起点和终点（在模型表面上）
    start_point = (78.56310916, 29.99410818, -162.10114156)
    goal_point = (83.32050134, 42.84368528, -164.97868412)

    print(f"原始起点: {start_point}")
    print(f"原始终点: {goal_point}")

    # 执行修改版A*路径规划，处理表面上的点
    path = astar_3d_surface(mesh, start_point, goal_point, occupied_voxels, voxel_size, bounds)
    process_path = post_process_path(mesh, path, start_point, goal_point)
    # 输出路径坐标
    if path:
        print("\n路径坐标:")
        for i, point in enumerate(path):
            print(f"点 {i}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")

        # 如果需要，可以将路径保存到文件
        with open("path.txt", "w") as f:
            for point in path:
                f.write(f"{point[0]:.4f} {point[1]:.4f} {point[2]:.4f}\n")
        print("路径已保存到path.txt")

    if process_path:
        print("\n路径坐标:")
        for i, point in enumerate(process_path):
            print(f"点 {i}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")

    # 可选：生成可视化文件
    try:
        PathVisualizer.visualize_with_open3d(mesh, start_point, goal_point, process_path)
    except Exception as e:
        print(f"无法生成可视化: {e}")


if __name__ == "__main__":
    main()
