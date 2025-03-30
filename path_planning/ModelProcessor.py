import trimesh


# 模型加载和处理模块
class ModelProcessor:
    def __init__(self, mesh=None, voxel_size=0.1, voxel_visualize=False):
        self.mesh = mesh
        self.voxel_size = voxel_size
        self.voxel_visualize = voxel_visualize

    def load_model(self, obj_file):
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

        self.mesh = mesh

        return mesh

    def voxelize_mesh(self):
        """将网格体素化"""
        print(f"开始体素化...")

        # 计算模型边界
        bounds = self.mesh.bounds
        print(f"模型边界: {bounds}")
        print(f"模型尺寸: {bounds[1] - bounds[0]}")

        # 体素化网格
        voxel_grid = self.mesh.voxelized(pitch=self.voxel_size)

        if self.voxel_visualize:
            # 将体素转换为立方体网格并可视化
            boxes_mesh = voxel_grid.as_boxes()
            boxes_mesh.show()

        occupied_voxels = set(map(tuple, voxel_grid.points))
        print(f"生成了 {len(occupied_voxels)} 个占用体素")

        return occupied_voxels, bounds
