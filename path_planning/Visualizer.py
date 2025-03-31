import numpy as np
import open3d as o3d
import trimesh


# 可视化模块
class Visualizer:

    def visualize_with_open3d(self, mesh, start, end, path):
        """使用Open3D可视化3D模型和路径"""
        # 将Trimesh转换为Open3D网格
        print("通过open3d进行可视化")
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.compute_vertex_normals()

        # 创建起点和终点球体
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        start_sphere.translate(start)
        start_sphere.paint_uniform_color([0, 0, 1])  # 蓝色

        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
        end_sphere.translate(end)
        end_sphere.paint_uniform_color([0, 1, 0])  # 绿色

        # 根据original_path创建路径点的小球
        path_spheres = []
        for point in path:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.translate(point)
            sphere.paint_uniform_color([1, 0, 0])  # 红色
            path_spheres.append(sphere)

        # 创建线段连接路径点
        lines = []
        points = []
        for i in range(len(path)):
            points.append(path[i])

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

    def visualize_with_trimesh(self, mesh, start, end, path):
        """
        使用trimesh可视化3D路径规划结果

        参数:
        mesh: trimesh.Trimesh对象，表示3D环境
        start: 起点坐标，形如[x, y, z]
        end: 终点坐标，形如[x, y, z]
        path: 路径点列表，每个元素形如[x, y, z]
        """
        print("通过trimesh进行可视化")

        # 创建场景
        scene = trimesh.Scene()

        # 添加网格到场景
        scene.add_geometry(mesh)

        # 创建起点球体（绿色）
        start_sphere = trimesh.primitives.Sphere(radius=0.1, center=start)
        start_sphere.visual.face_colors = [0, 255, 0, 255]  # 绿色
        scene.add_geometry(start_sphere)

        # 创建终点球体（红色）
        end_sphere = trimesh.primitives.Sphere(radius=0.1, center=end)
        end_sphere.visual.face_colors = [255, 0, 0, 255]  # 红色
        scene.add_geometry(end_sphere)

        # 创建路径点（蓝色小球）
        for point in path:
            path_sphere = trimesh.primitives.Sphere(radius=0.05, center=point)
            path_sphere.visual.face_colors = [0, 0, 255, 255]  # 蓝色
            scene.add_geometry(path_sphere)

        # 创建路径线
        if len(path) > 0:
            # 将起点、路径点和终点连接起来
            all_points = np.vstack([[start], path, [end]])

            # 创建线段列表
            lines = []
            for i in range(len(all_points) - 1):
                lines.append([i, i + 1])

            # 创建路径线对象
            path_line = trimesh.path.Path3D(
                entities=[trimesh.path.entities.Line(points=np.arange(len(all_points)))],
                vertices=all_points)

            # 设置颜色 - 每个实体一个颜色
            path_line.colors = np.array([[255, 255, 0, 255]])  # 黄色路径线

            scene.add_geometry(path_line)

        # 显示场景
        scene.show()


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
