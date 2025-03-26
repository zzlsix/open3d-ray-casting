import numpy as np


# 路径平滑器类
class PathSmoother:
    def __init__(self, mesh, collision_checker, iterations=2):
        self.mesh = mesh
        self.collision_checker = collision_checker
        self.iterations = iterations

    def smooth_path(self, path):
        """
        使用移动平均和直线可行性检查来平滑路径
        """
        if len(path) <= 2:
            return path

        smoothed = path.copy()

        for _ in range(self.iterations):
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
                    if self.collision_checker.ray_collision_check(avg, neighbor):
                        collision = True
                        break

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
                        if not self.collision_checker.ray_collision_check(smoothed[i], smoothed[i + 2]):
                            # 可以安全地移除中间点
                            smoothed.pop(i + 1)
                            continue
                i += 1

        return smoothed