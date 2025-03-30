from path_planning.ModelProcessor import ModelProcessor
from path_planning.PathPlanner import PathPlanner
from path_planning.Visualizer import Visualizer
from path_planning.algorithm.AStarAlgorithm import AStarAlgorithm
from path_planning.algorithm.RRTAlgorithm import RRTAlgorithm


# 应用主控制器
class PathPlanningApp:

    def run(self, obj_file, start_point, goal_point, algorithm):
        """执行完整的路径规划过程"""

        model_processor = ModelProcessor()
        # 加载并处理模型
        mesh = model_processor.load_model(obj_file)
        # 体素化模型
        voxel_grid = model_processor.voxelize_mesh(mesh)

        # 4. 创建路径规划器并寻找路径
        path_planner = PathPlanner(mesh, voxel_grid, model_processor.voxel_size, algorithm)

        print(f"原始起点: {start_point}")
        print(f"原始终点: {goal_point}")

        path = path_planner.find_path(start_point, goal_point)

        if path:
            visualizer = Visualizer()
            visualizer.print_path(path)
            visualizer.save_path_to_file(path)

            # 7. 可视化
            try:
                visualizer.visualize_with_open3d(mesh, start_point, goal_point, path)
            except Exception as e:
                print(f"无法生成可视化: {e}")

            return path
        else:
            print("无法找到有效路径")
            return None


if __name__ == "__main__":
    # 文件路径
    obj_file = r"C:\Users\mi\Downloads\Three-Dimension-3D\models\pc\0\terra_obj\Block\Block.obj"

    # 用户提供的起点和终点（在模型表面上）
    start_point = (78.56310916, 29.99410818, -162.10114156)
    goal_point = (83.32050134, 42.84368528, -164.97868412)

    # 创建并运行应用
    app = PathPlanningApp()
    app.run(obj_file, start_point, goal_point, AStarAlgorithm())
