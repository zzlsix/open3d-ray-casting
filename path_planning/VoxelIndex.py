import numpy as np


class VoxelIndex:
    def __init__(self, voxel_grid):
        self.origin = voxel_grid.bounds[0]
        self.pitch = voxel_grid.pitch
        # 预处理：构建占用体素索引集合
        self.occupied_indices = set()
        for point in voxel_grid.points:
            index = tuple(np.floor((point - self.origin) / self.pitch).astype(int))
            self.occupied_indices.add(index)
        print("成功构建体素索引")

    def is_occupied(self, point):
        """
        判断世界坐标的点是否在体素中
        :param point: 世界坐标点
        """
        # 将点转换为体素索引
        index = tuple(np.floor((np.array(point) - self.origin) / self.pitch).astype(int))
        # 检查索引是否在占用集合中
        return index in self.occupied_indices