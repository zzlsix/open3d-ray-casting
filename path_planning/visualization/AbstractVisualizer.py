from enum import Enum

import numpy as np


class AbstractVisualizer:
    def show(self, mesh, start, end, path, **kwargs):
        pass

    @staticmethod
    def save_path_to_file(path, filename="path.txt"):
        """将路径保存到文件"""
        with open(filename, "w") as f:
            for point in path:
                f.write(f"{point[0]:.4f} {point[1]:.4f} {point[2]:.4f}\n")
        print(f"路径已保存到{filename}")

    @staticmethod
    def print_path(path):
        """打印路径坐标"""
        print("\n路径坐标:")
        for i, point in enumerate(path):
            print(f"点 {i}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")


class Open3dColor(Enum):
    # 基础颜色
    RED = (1.0, 0.0, 0.0)
    GREEN = (0.0, 1.0, 0.0)
    BLUE = (0.0, 0.0, 1.0)
    YELLOW = (1.0, 1.0, 0.0)
    CYAN = (0.0, 1.0, 1.0)
    MAGENTA = (1.0, 0.0, 1.0)
    WHITE = (1.0, 1.0, 1.0)
    BLACK = (0.0, 0.0, 0.0)
    GRAY = (0.5, 0.5, 0.5)

    # 扩展颜色
    ORANGE = (1.0, 0.5, 0.0)
    PURPLE = (0.5, 0.0, 0.5)
    BROWN = (0.5, 0.25, 0.0)
    PINK = (1.0, 0.75, 0.8)
    LIGHT_BLUE = (0.5, 0.8, 1.0)
    DARK_GREEN = (0.0, 0.5, 0.0)
    DARK_RED = (0.5, 0.0, 0.0)
    GOLD = (1.0, 0.84, 0.0)
    SILVER = (0.75, 0.75, 0.75)