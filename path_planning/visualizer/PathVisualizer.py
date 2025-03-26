import matplotlib.pyplot as plt
import numpy as np


# 可视化器类
class PathVisualizer:
    @staticmethod
    def visualize_no_path(mesh, start, end, connected_components, vertices):
        """可视化网格、起点、终点和连通分量"""
        print("可视化网格和连通分量...")
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制起点和终点
        ax.scatter([start[0]], [start[1]], [start[2]], color='blue', s=100, label='起点')
        ax.scatter([end[0]], [end[1]], [end[2]], color='green', s=100, label='终点')

        # 绘制不同连通分量的部分顶点
        colors = plt.cm.rainbow(np.linspace(0, 1, len(connected_components)))
        for i, component in enumerate(connected_components):
            # 从每个分量中取样一些点来可视化
            sample_size = min(100, len(component))
            sample = np.random.choice(list(component), sample_size, replace=False)
            component_vertices = vertices[sample]
            ax.scatter(
                component_vertices[:, 0],
                component_vertices[:, 1],
                component_vertices[:, 2],
                color=colors[i],
                alpha=0.5,
                s=10,
                label=f'组件 {i} ({len(component)} 个点)'
            )

        ax.set_title('3D模型 - 无法找到路径')
        ax.set_xlabel('X轴')
        ax.set_ylabel('Y轴')
        ax.set_zlabel('Z轴')
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_with_open3d(mesh, start, end, original_path, smoothed_path=None, collision_points=None):
        """使用Open3D可视化3D模型和路径"""
        try:
            import open3d as o3d
        except ImportError:
            print("需要安装Open3D: pip install open3d")
            return

        # 将Trimesh转换为Open3D网格
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.compute_vertex_normals()

        # 创建原始路径的线集合
        path_array = np.array(original_path)
        path_lines = o3d.geometry.LineSet()
        path_lines.points = o3d.utility.Vector3dVector(path_array)
        path_lines.lines = o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(path_array) - 1)]
        )
        path_lines.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(path_array) - 1)])

        # 创建平滑路径的线集合
        smooth_lines = None
        if smoothed_path is not None and id(smoothed_path) != id(original_path):
            smooth_array = np.array(smoothed_path)
            smooth_lines = o3d.geometry.LineSet()
            smooth_lines.points = o3d.utility.Vector3dVector(smooth_array)
            smooth_lines.lines = o3d.utility.Vector2iVector(
                [[i, i + 1] for i in range(len(smooth_array) - 1)]
            )
            smooth_lines.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(smooth_array) - 1)])

        # 创建起点和终点球体
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        start_sphere.translate(start)
        start_sphere.paint_uniform_color([0, 0, 1])  # 蓝色

        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        end_sphere.translate(end)
        end_sphere.paint_uniform_color([0, 1, 0])  # 绿色

        # 创建路径点的小球
        path_spheres = []
        for i in range(0, len(path_array), max(1, len(path_array) // 20)):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
            sphere.translate(path_array[i])
            sphere.paint_uniform_color([1, 1, 0])
            path_spheres.append(sphere)

        # 显示所有几何体
        geometries = [o3d_mesh, path_lines, start_sphere, end_sphere] + path_spheres
        if smooth_lines is not None:
            geometries.append(smooth_lines)

        # 如果有碰撞点，显示它们
        if collision_points and len(collision_points) > 0:
            for i, start_pt, end_pt in collision_points:
                # 创建红色球体标记碰撞点
                collision_sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.8)
                collision_sphere1.translate(start_pt)
                collision_sphere1.paint_uniform_color([1, 0, 0])  # 红色

                collision_sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.8)
                collision_sphere2.translate(end_pt)
                collision_sphere2.paint_uniform_color([1, 0, 0])  # 红色

                # 添加到几何体列表
                geometries.extend([collision_sphere1, collision_sphere2])

        # 设置渲染选项并显示
        o3d.visualization.draw_geometries(
            geometries,
            window_name="3D Model Path Planning",
            width=1024,
            height=768,
            point_show_normal=False,
            mesh_show_wireframe=True,
            mesh_show_back_face=False,
        )