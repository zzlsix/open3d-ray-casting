from path_planning.ModelProcessor import ModelProcessor
from path_planning.PathPlanner import PathPlanner
from path_planning.Visualizer import Visualizer
from path_planning.algorithm.RRTAlgorithm import RRTAlgorithm


# 应用主控制器
class PathPlanningApp:
    def __init__(self, obj_file):
        self.obj_file = obj_file
        self.model_processor = ModelProcessor()
        self.visualizer = Visualizer()

    def run(self, start_point, goal_point):
        """执行完整的路径规划过程"""
        # 1. 加载并处理模型
        mesh = self.model_processor.load_model(self.obj_file)

        # 2. 估计合适的体素大小
        voxel_size = self.model_processor.estimate_voxel_size(mesh)

        # 3. 体素化模型
        occupied_voxels, bounds = self.model_processor.voxelize_mesh(mesh, voxel_size)

        # 4. 创建路径规划器并寻找路径
        path_planner = PathPlanner(mesh, occupied_voxels, voxel_size, bounds, RRTAlgorithm())

        print(f"原始起点: {start_point}")
        print(f"原始终点: {goal_point}")

        path = path_planner.find_path(start_point, goal_point)

        if path:
            # 5. 后处理路径，使其更靠近表面
            processed_path = path_planner.post_process_path(path, start_point, goal_point)

            # 6. 显示和保存结果
            self.visualizer.print_path(processed_path)
            self.visualizer.save_path_to_file(processed_path)

            # 7. 可视化
            try:
                self.visualizer.visualize_with_open3d(mesh, start_point, goal_point, processed_path)
            except Exception as e:
                print(f"无法生成可视化: {e}")

            return processed_path
        else:
            print("无法找到有效路径")
            return None


def main():
    # 配置参数
    obj_file = r"C:\Users\mi\Downloads\Three-Dimension-3D\models\pc\0\terra_obj\Block\Block.obj"  # 替换为你的OBJ文件路径

    # 用户提供的起点和终点（在模型表面上）
    start_point = (78.56310916, 29.99410818, -162.10114156)
    goal_point = (83.32050134, 42.84368528, -164.97868412)

    # 创建并运行应用
    app = PathPlanningApp(obj_file)
    app.run(start_point, goal_point)


if __name__ == "__main__":
    main()
