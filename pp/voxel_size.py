import numpy as np

"""
应用场景	               体素尺寸比例	                       说明
高精度碰撞检测	模型直径/200 - 模型直径/500	用于精确的障碍物规避，但计算成本高
标准路径规划	    模型直径/100 - 模型直径/200	    平衡精度和性能的一般选择
粗略可视化规划	模型直径/50 - 模型直径/100	        快速预览，适合交互式规划
大型环境快速规划	模型直径/20 - 模型直径/50	        适用于地形或建筑物等大型模型
"""

def calculate_appropriate_voxel_size(mesh, path_precision_factor=20, memory_limit_gb=4):
    """
    计算合适的体素尺寸

    参数:
        mesh: Open3D网格
        path_precision_factor: 路径精度因子(越大体素越小)
        memory_limit_gb: 最大允许内存使用量(GB)

    返回:
        推荐的体素尺寸
    """
    # 1. 基于模型尺寸计算
    vertices = np.asarray(mesh.vertices)
    min_bound = np.min(vertices, axis=0)
    max_bound = np.max(vertices, axis=0)

    # 计算模型直径
    model_diameter = np.linalg.norm(max_bound - min_bound)
    size_based_voxel = model_diameter / 100

    # 2. 基于特征尺寸计算 (估计)
    # 这里使用边长分布来估计特征尺寸
    triangles = np.asarray(mesh.triangles)
    edges = []
    for triangle in triangles[:min(10000, len(triangles))]:  # 采样部分三角形以提高速度
        v0, v1, v2 = vertices[triangle]
        edges.append(np.linalg.norm(v1 - v0))
        edges.append(np.linalg.norm(v2 - v1))
        edges.append(np.linalg.norm(v0 - v2))

    # 使用边长分布的低百分位数作为特征尺寸估计
    feature_size = np.percentile(edges, 10)  # 使用第10百分位
    feature_based_voxel = feature_size / 2.5

    # 3. 基于路径规划精度
    # 假设需要path_precision_factor个体素来表示模型的典型尺寸
    precision_based_voxel = model_diameter / path_precision_factor

    # 4. 基于内存限制计算
    # 估计体素数量和内存使用量
    def estimate_voxel_count(voxel_size):
        grid_resolution = (max_bound - min_bound) / voxel_size
        return np.prod(np.ceil(grid_resolution))

    def estimate_memory_usage_gb(voxel_count):
        # 每个体素约占用32字节(位置、索引和其他开销)
        return (voxel_count * 32) / (1024 ** 3)

    # 二分查找满足内存限制的最小体素尺寸
    min_size = max(size_based_voxel, feature_based_voxel, precision_based_voxel) / 10
    max_size = model_diameter / 10

    while max_size - min_size > 0.001 * model_diameter:
        mid_size = (min_size + max_size) / 2
        voxel_count = estimate_voxel_count(mid_size)
        memory_usage = estimate_memory_usage_gb(voxel_count)

        if memory_usage <= memory_limit_gb:
            max_size = mid_size
        else:
            min_size = mid_size

    memory_based_voxel = max_size

    # 5. 综合考虑各因素，取最大值以确保性能
    recommended_voxel_size = max(
        size_based_voxel,
        feature_based_voxel,
        precision_based_voxel,
        memory_based_voxel
    )

    # 打印各种计算结果供参考
    print(f"模型直径: {model_diameter:.4f}")
    print(f"估计特征尺寸: {feature_size:.4f}")
    print(f"基于尺寸推荐: {size_based_voxel:.4f}")
    print(f"基于特征推荐: {feature_based_voxel:.4f}")
    print(f"基于精度推荐: {precision_based_voxel:.4f}")
    print(f"基于内存推荐: {memory_based_voxel:.4f}")
    print(f"最终推荐体素尺寸: {recommended_voxel_size:.4f}")

    return recommended_voxel_size