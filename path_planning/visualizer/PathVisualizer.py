import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


# 可视化器类：用于将路径规划结果以图形方式展示
class PathVisualizer:
    @staticmethod
    def visualize_no_path(mesh, start, end, connected_components, vertices):
        """
        当无法找到路径时，可视化网格、起点、终点和连通分量

        参数:
            mesh: 3D网格模型
            start: 起点坐标 [x,y,z]
            end: 终点坐标 [x,y,z]
            connected_components: 网格顶点的连通分量列表，每个分量是顶点索引的集合
            vertices: 网格的所有顶点坐标数组
        """
        print("可视化网格和连通分量...")
        fig = plt.figure(figsize=(12, 8))  # 创建大小为12x8的图形
        ax = fig.add_subplot(111, projection='3d')  # 创建3D子图

        # 绘制起点和终点：蓝色为起点，绿色为终点
        ax.scatter([start[0]], [start[1]], [start[2]], color='blue', s=100, label='起点')
        ax.scatter([end[0]], [end[1]], [end[2]], color='green', s=100, label='终点')

        # 绘制不同连通分量的部分顶点，使用彩虹色谱区分不同分量
        colors = plt.cm.rainbow(np.linspace(0, 1, len(connected_components)))  # 生成彩虹色谱
        for i, component in enumerate(connected_components):
            # 从每个分量中取样最多100个点来可视化，避免绘制过多点导致图形卡顿
            sample_size = min(100, len(component))
            sample = np.random.choice(list(component), sample_size, replace=False)
            component_vertices = vertices[sample]  # 获取采样点的实际坐标
            ax.scatter(
                component_vertices[:, 0],  # X坐标
                component_vertices[:, 1],  # Y坐标
                component_vertices[:, 2],  # Z坐标
                color=colors[i],  # 使用对应的颜色
                alpha=0.5,  # 设置透明度为0.5
                s=10,  # 点大小为10
                label=f'组件 {i} ({len(component)} 个点)'  # 标签显示组件编号和点数
            )

        # 设置图表标题和轴标签
        ax.set_title('3D模型 - 无法找到路径')
        ax.set_xlabel('X轴')
        ax.set_ylabel('Y轴')
        ax.set_zlabel('Z轴')
        plt.legend()  # 显示图例
        plt.tight_layout()  # 自动调整布局
        plt.show()  # 显示图形

    @staticmethod
    def visualize_with_open3d(mesh, start, end, original_path, smoothed_path=None, collision_points=None):
        """
        使用Open3D库可视化3D模型和路径

        参数:
            mesh: 3D网格模型
            start: 起点坐标
            end: 终点坐标
            original_path: 原始规划路径
            smoothed_path: 平滑后的路径(可选)
            collision_points: 碰撞检测点列表(可选)，包含索引和碰撞点坐标
        """

        # 将Trimesh格式的网格转换为Open3D格式
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)  # 设置顶点
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)  # 设置三角面
        o3d_mesh.compute_vertex_normals()  # 计算顶点法线，用于光照效果

        # 创建起点和终点的球体标记
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
        start_sphere.translate(start)
        start_sphere.paint_uniform_color([0, 1, 0])  # 绿色起点

        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
        end_sphere.translate(end)
        end_sphere.paint_uniform_color([0, 0, 1])  # 蓝色终点

        # 创建路径点的球体标记
        path_spheres = []
        for point in original_path:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
            sphere.translate(point)
            sphere.paint_uniform_color([1, 0, 0])  # 红色路径点
            path_spheres.append(sphere)

        # 显示所有几何体，但不连接路径点
        geometries = [o3d_mesh, start_sphere, end_sphere]
        geometries.extend(path_spheres)

        # 打开可视化窗口
        o3d.visualization.draw_geometries(geometries,
                                          window_name="3D Path Visualization",
                                          width=1024,
                                          height=768,
                                          point_show_normal=False)