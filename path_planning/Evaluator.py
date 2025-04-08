import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

from PathPlanningApp import PathPlanningApp
from path_planning.ModelProcessor import ModelProcessor
from path_planning.PathPlanner import PathPlanner


class Evaluator:
    def __init__(self):
        self.results = {}

    def evaluate_all_algorithms(self, obj_file, start_point, goal_point, show_process=False):
        app = PathPlanningApp()
        algorithms = ["A*", "RRT", "RRT*"]

        model_processor = ModelProcessor()
        mesh = model_processor.load_model(obj_file)
        voxel_grid = model_processor.voxelize_mesh(mesh)

        for algo in algorithms:
            print(f"\n评估算法: {algo}")
            path, metrics = self.evaluate_algorithm(app,
                                                    mesh,
                                                    voxel_grid,
                                                    model_processor.voxel_size,
                                                    start_point,
                                                    goal_point,
                                                    algo,
                                                    show_process=False)
            self.results[algo] = {
                'path': path,
                'metrics': metrics
            }

        self.compare_results()

    def evaluate_algorithm(self, app, mesh, voxel_grid, voxel_size, start_point, goal_point, algorithm, show_process):
        # 记录开始时间
        start_time = time.time()

        # 加载模型处理器以获取网格信息

        # 创建路径规划器
        path_planner = PathPlanner(mesh, voxel_grid, voxel_size, app.algorithms[algorithm])

        # 寻找路径并记录内存使用
        import tracemalloc
        tracemalloc.start()
        path, algo_time = path_planner.find_path(start_point, goal_point)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 计算总时间
        total_time = time.time() - start_time

        # 计算路径指标
        metrics = {}

        if path:
            # 路径长度
            path_length = self.calculate_path_length(path)

            # 路径平滑度 (通过计算连续三点形成的角度变化)
            smoothness = self.calculate_smoothness(path)

            # 安全裕度 (路径与障碍物的最小距离)
            safety_margin = self.calculate_safety_margin(path, voxel_grid, voxel_size)

            # 节点数量
            node_count = len(path)

            metrics = {
                'success': True,
                'path_length': path_length,
                'smoothness': smoothness,
                'safety_margin': safety_margin,
                'computation_time': algo_time,
                'total_time': total_time,
                'memory_usage_kb': peak / 1024,
                'node_count': node_count
            }
        else:
            metrics = {
                'success': False,
                'computation_time': algo_time,
                'total_time': total_time,
                'memory_usage_kb': peak / 1024
            }

        print(f"算法: {algorithm}")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        return path, metrics

    def calculate_path_length(self, path):
        """计算路径总长度"""
        length = 0
        for i in range(1, len(path)):
            length += euclidean(path[i - 1], path[i])
        return length

    def calculate_smoothness(self, path):
        """计算路径平滑度 (角度变化的平均值)"""
        if len(path) < 3:
            return 1.0  # 完全平滑

        angle_changes = []
        for i in range(1, len(path) - 1):
            v1 = np.array(path[i]) - np.array(path[i - 1])
            v2 = np.array(path[i + 1]) - np.array(path[i])

            # 计算向量范数
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            # 检查向量是否为零向量
            if norm_v1 < 1e-10 or norm_v2 < 1e-10:
                continue  # 跳过这个点

            # 计算向量夹角
            cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免数值误差
            angle = np.arccos(cos_angle)
            angle_changes.append(angle)

        # 如果没有有效的角度变化，返回完全平滑
        if not angle_changes:
            return 1.0

        # 平滑度指标: 1 - 平均角度变化/π (值越接近1表示越平滑)
        smoothness = 1 - np.mean(angle_changes) / np.pi
        return smoothness

    def calculate_safety_margin(self, path, voxel_grid, voxel_size):
        """计算路径与障碍物之间的最小距离"""
        # 简化实现，实际应用中可能需要更复杂的计算
        min_distance = float('inf')

        # 这里需要根据你的voxel_grid实现来计算
        # 简单示例:
        for point in path:
            # 找到最近的障碍物
            # distance = min_distance_to_obstacle(point, voxel_grid, voxel_size)
            # min_distance = min(min_distance, distance)
            pass

        return min_distance if min_distance != float('inf') else 0

    def compare_results(self):
        """比较不同算法的结果并可视化"""
        if not self.results:
            print("没有可比较的结果")
            return

        # 提取指标进行比较
        algorithms = list(self.results.keys())
        metrics = ['path_length', 'smoothness', 'computation_time', 'memory_usage_kb', 'node_count']

        # 创建比较表格
        print("\n算法比较结果:")
        print("-" * 80)
        header = "target".ljust(20)
        for algo in algorithms:
            header += algo.ljust(20)
        print(header)
        print("-" * 80)

        for metric in metrics:
            row = metric.ljust(20)
            for algo in algorithms:
                if metric in self.results[algo]['metrics']:
                    value = self.results[algo]['metrics'][metric]
                    row += f"{value:.4f}".ljust(20)
                else:
                    row += "N/A".ljust(20)
            print(row)

        # 计算综合评分
        scores, normalized_data = self.calculate_overall_score()

        # 打印综合评分
        print("\n算法综合评分:")
        print("-" * 40)
        for algo, score in scores.items():
            print(f"{algo}: {score:.4f}")

        # 可视化比较
        self.visualize_comparison()

        # 雷达图分析
        self.visualize_radar_chart(normalized_data)

    def visualize_comparison(self):
        """可视化不同算法的性能比较"""
        if not self.results:
            return

        algorithms = list(self.results.keys())

        # 比较计算时间
        times = [self.results[algo]['metrics'].get('computation_time', 0) for algo in algorithms]

        plt.figure(figsize=(12, 6))

        # 计算时间比较
        plt.subplot(1, 2, 1)
        plt.bar(algorithms, times)
        plt.title('comparison of calculation time')
        plt.ylabel('time(seconds)')

        # 路径长度比较
        if all('path_length' in self.results[algo]['metrics'] for algo in algorithms):
            lengths = [self.results[algo]['metrics']['path_length'] for algo in algorithms]
            plt.subplot(1, 2, 2)
            plt.bar(algorithms, lengths)
            plt.title('path length comparison')
            plt.ylabel('length')

        plt.tight_layout()
        plt.savefig('algorithm_comparison.png')
        plt.show()

    def calculate_overall_score(self):
        """计算各算法的综合评分"""
        if not self.results:
            return {}

        # 提取所有算法的指标值
        metrics_data = {}
        for metric in ['path_length', 'computation_time', 'smoothness', 'safety_margin', 'memory_usage_kb']:
            metrics_data[metric] = []
            for algo in self.results:
                if metric in self.results[algo]['metrics']:
                    metrics_data[metric].append(self.results[algo]['metrics'][metric])

        # 归一化处理
        normalized_data = {}
        for algo in self.results:
            normalized_data[algo] = {}
            for metric in metrics_data:
                if metric not in self.results[algo]['metrics']:
                    continue

                value = self.results[algo]['metrics'][metric]
                min_val = min(metrics_data[metric])
                max_val = max(metrics_data[metric])

                # 避免除零错误
                if max_val == min_val:
                    normalized_data[algo][metric] = 1.0
                    continue

                # 根据指标类型选择归一化方向
                if metric in ['smoothness', 'safety_margin']:
                    # 这些指标越大越好
                    normalized_data[algo][metric] = (value - min_val) / (max_val - min_val)
                else:
                    # 这些指标越小越好
                    normalized_data[algo][metric] = (max_val - value) / (max_val - min_val)

        # 定义权重
        weights = {
            'path_length': 0.25,
            'computation_time': 0.20,
            'smoothness': 0.20,
            'safety_margin': 0.25,
            'memory_usage_kb': 0.10
        }

        # 计算加权得分
        scores = {}
        for algo in normalized_data:
            score = 0
            for metric in normalized_data[algo]:
                score += normalized_data[algo][metric] * weights[metric]
            scores[algo] = score

        return scores, normalized_data

    def visualize_radar_chart(self, normalized_data):
        """使用雷达图可视化各算法在不同指标上的表现"""
        algorithms = list(normalized_data.keys())
        metrics = ['path_length', 'computation_time', 'smoothness', 'safety_margin', 'memory_usage_kb']

        # 准备雷达图数据
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

        for algo in algorithms:
            values = []
            for metric in metrics:
                if metric in normalized_data[algo]:
                    values.append(normalized_data[algo][metric])
                else:
                    values.append(0)
            values += values[:1]  # 闭合雷达图

            ax.plot(angles, values, linewidth=2, label=algo)
            ax.fill(angles, values, alpha=0.1)

        # 设置雷达图标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.set_ylim(0, 1)

        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('comparison of algorithm performance radar graphs')
        plt.tight_layout()
        plt.savefig('algorithm_radar_comparison.png')
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 文件路径
    obj_file = r"C:\Users\mi\Downloads\Three-Dimension-3D\models\pc\0\terra_obj\Block\Block.obj"

    # 起点和终点
    start_point = (78.56310916, 29.99410818, -162.10114156)
    goal_point = (83.32050134, 42.84368528, -164.97868412)

    # 创建评估器并评估所有算法
    evaluator = Evaluator()
    evaluator.evaluate_all_algorithms(obj_file, start_point, goal_point, show_process=False)
