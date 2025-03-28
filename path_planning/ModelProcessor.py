import trimesh


# 模型加载和处理模块
class ModelProcessor:
    @staticmethod
    def load_model(obj_file):
        """加载3D模型文件"""
        print(f"加载模型: {obj_file}")
        model = trimesh.load(obj_file)

        # 处理Scene对象
        if isinstance(model, trimesh.Scene):
            print("检测到Scene对象，合并所有网格...")
            geometries = list(model.geometry.values())
            if not geometries:
                raise ValueError("无法从模型中提取几何体")

            mesh = geometries[0] if len(geometries) == 1 else trimesh.util.concatenate(geometries)
        else:
            mesh = model

        return mesh

    @staticmethod
    def estimate_voxel_size(mesh):
        """估计合适的体素大小"""
        bounds = mesh.bounds
        dimensions = bounds[1] - bounds[0]

        # 取模型最小尺寸的1%作为体素大小
        suggested_size = min(dimensions) * 0.01

        # 设置上下限
        min_size, max_size = 0.01, 1.0
        voxel_size = max(min_size, min(max_size, suggested_size))

        print(f"模型尺寸: {dimensions}")
        print(f"建议的体素大小: {voxel_size}")

        return voxel_size

    @staticmethod
    def voxelize_mesh(mesh, voxel_size):
        """将网格体素化"""
        print(f"开始体素化...")

        # 计算模型边界
        bounds = mesh.bounds
        print(f"模型边界: {bounds}")
        print(f"模型尺寸: {bounds[1] - bounds[0]}")

        # 体素化网格
        voxel_grid = mesh.voxelized(pitch=voxel_size)
        occupied_voxels = set(map(tuple, voxel_grid.points))
        print(f"生成了 {len(occupied_voxels)} 个占用体素")

        return occupied_voxels, bounds
