import numpy as np


class CollisionChecker:
    def __init__(self, mesh, safety_margin=0.01):
        self.mesh = mesh
        self.safety_margin = safety_margin

    def ray_collision_check(self, start_point, end_point):
        """检查路径段是否与网格碰撞"""
        direction = end_point - start_point
        distance = np.linalg.norm(direction)

        if distance < 1e-6:  # 防止距离过小导致的问题
            return False

        direction = direction / distance

        # 在起点略微偏移以避免自相交
        offset_start = start_point + direction * self.safety_margin

        # 设置射线长度略小于总距离，避免终点处的误检
        ray_length = distance - 2 * self.safety_margin

        if ray_length <= 0:
            return False  # 太短的线段视为无碰撞

        """
        起点 ●───●───●───●───●───●───●───● 终点
            |   |   |   |   |   |   |   |
            |   |   |   |   |   |   |   |
            ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
            射线检测片段 (每段单独进行碰撞检测)
        """
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
                        hits = self.mesh.ray.intersects_any(
                            ray_origins=[ray_origin],
                            ray_directions=[sub_direction],
                            ray_distances=[sub_length]
                        )
                        if any(hits):
                            return True  # 检测到碰撞
                    except (AttributeError, ValueError) as e:
                        print(f"射线检测异常: {e}")

        return False  # 未检测到碰撞