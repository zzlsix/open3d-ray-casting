import numpy as np
import trimesh


# 几何工具模块
class GeometryTools:
    @staticmethod
    def is_point_inside_mesh(mesh, point):
        """检查点是否在网格内部"""
        ray_origins = np.array([point])
        ray_directions = np.array([[1.0, 0.0, 0.0]])  # 沿x轴正方向

        # 计算射线与网格的交点数
        intersections = mesh.ray.intersects_location(ray_origins, ray_directions)

        # 如果交点数为奇数，则点在内部
        return len(intersections[0]) % 2 == 1

    @staticmethod
    def find_nearest_surface_point(mesh, point):
        """找到距离给定点最近的表面点，确保在表面外部"""
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
            offset_distance = 1
            safe_point = point + normal * offset_distance
            print(f"表面点已偏移: 从 {point} 到 {safe_point}")
            return safe_point

        # 如果点不在表面上但靠近内部
        if GeometryTools.is_point_inside_mesh(mesh, point):
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
