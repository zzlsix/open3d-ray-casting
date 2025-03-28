import numpy as np
import open3d as o3d


# 可视化模块
class Visualizer:
    @staticmethod
    def visualize_with_open3d(mesh, start, end, original_path):
        """使用Open3D可视化3D模型和路径"""
        # 将Trimesh转换为Open3D网格
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.compute_vertex_normals()

        # 创建起点和终点球体
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        start_sphere.translate(start)
        start_sphere.paint_uniform_color([0, 0, 1])  # 蓝色

        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        end_sphere.translate(end)
        end_sphere.paint_uniform_color([0, 1, 0])  # 绿色

        # 根据original_path创建路径点的小球
        path_spheres = []
        for point in original_path:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            sphere.translate(point)
            sphere.paint_uniform_color([1, 0, 0])  # 红色
            path_spheres.append(sphere)

        # 创建线段连接路径点
        lines = []
        points = []
        for i in range(len(original_path)):
            points.append(original_path[i])

        if len(points) > 1:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_indices = [[i, i + 1] for i in range(len(points) - 1)]
            line_set.lines = o3d.utility.Vector2iVector(line_indices)
            line_set.colors = o3d.utility.Vector3dVector([[1, 0.5, 0]] * len(line_indices))  # 橙色
            lines.append(line_set)

        # 将所有几何体添加到列表中
        geometries = [o3d_mesh, start_sphere, end_sphere] + path_spheres + lines

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
