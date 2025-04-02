import time

from path_planning.visualization.AbstractVisualizer import AbstractVisualizer
import numpy as np
import open3d as o3d

class ProcessVisualizer(AbstractVisualizer):
    def _trimesh_to_open3d(self, trimesh_mesh):
        """将trimesh网格转换为Open3D网格"""
        # 创建Open3D网格
        mesh_o3d = o3d.geometry.TriangleMesh()

        # 设置顶点和三角形
        mesh_o3d.vertices = o3d.utility.Vector3dVector(np.array(trimesh_mesh.vertices))
        mesh_o3d.triangles = o3d.utility.Vector3iVector(np.array(trimesh_mesh.faces))

        # 计算法线
        mesh_o3d.compute_vertex_normals()

        # 如果trimesh有颜色，也可以转换颜色
        if hasattr(trimesh_mesh, 'visual') and hasattr(trimesh_mesh.visual, 'vertex_colors'):
            colors = np.array(trimesh_mesh.visual.vertex_colors[:, :3]) / 255.0
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colors)
        else:
            # 设置默认颜色
            mesh_o3d.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色

        return mesh_o3d

class RRTProcessVisualizer(ProcessVisualizer):
    def __init__(self):
        self.vis_o3d = None
        self.path_lines = None # 存放结果路径
        self.tree_lines = None # 存放rrt搜索过程

    def show(self, mesh, start, end, path, key="value"):
        self._init_process_visualization(mesh, start, end)


    def _init_process_visualization(self, mesh, start, goal):
        """使用Open3D初始化过程可视化"""

        # 创建可视化窗口
        self.vis_o3d = o3d.visualization.Visualizer()
        self.vis_o3d.create_window(window_name="RRT 3D Path Planning", width=1024, height=768)

        # 导入环境模型 - 从planner的mesh属性获取
        if mesh is not None:
            # 转换trimesh到open3d
            o3d_mesh = self._trimesh_to_open3d(mesh)
            self.vis_o3d.add_geometry(o3d_mesh)
            print("成功导入环境模型到Open3D可视化器")
        else:
            print("警告: 无法从planner获取mesh模型")

        # 创建坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
        self.vis_o3d.add_geometry(coordinate_frame)

        # 创建起点和终点球体
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        start_sphere.translate(start)
        start_sphere.paint_uniform_color([0, 1, 0])  # 绿色
        self.vis_o3d.add_geometry(start_sphere)

        goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        goal_sphere.translate(goal)
        goal_sphere.paint_uniform_color([1, 0, 0])  # 红色
        self.vis_o3d.add_geometry(goal_sphere)

        # 创建树的线集合
        self.tree_lines = o3d.geometry.LineSet()
        self.vis_o3d.add_geometry(self.tree_lines)

        # 创建路径线集合
        self.path_lines = o3d.geometry.LineSet()
        self.vis_o3d.add_geometry(self.path_lines)

        # 设置视角
        self.vis_o3d.get_render_option().point_size = 5
        self.vis_o3d.get_render_option().line_width = 2.0
        self.vis_o3d.get_render_option().background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景

        # 设置相机位置
        view_control = self.vis_o3d.get_view_control()
        view_control.set_zoom(0.8)

        # 更新视图
        self.vis_o3d.poll_events()
        self.vis_o3d.update_renderer()

    def update(self, tree, new_node=None, nearest_node=None, path=None):
        """使用Open3D更新可视化"""

        # 更新树
        points = []
        lines = []
        line_count = 0

        for node, parent in tree.items():
            if parent is not None:
                points.extend([parent, node])
                lines.append([line_count, line_count + 1])
                line_count += 2

        if points:
            self.tree_lines.points = o3d.utility.Vector3dVector(points)
            self.tree_lines.lines = o3d.utility.Vector2iVector(lines)
            self.tree_lines.paint_uniform_color([0, 0, 1])  # 蓝色

        # 更新路径
        if path is not None:
            path_points = []
            path_lines = []

            for i in range(len(path) - 1):
                path_points.extend([path[i], path[i + 1]])
                path_lines.append([i * 2, i * 2 + 1])

            self.path_lines.points = o3d.utility.Vector3dVector(path_points)
            self.path_lines.lines = o3d.utility.Vector2iVector(path_lines)
            self.path_lines.paint_uniform_color([1, 0, 0])  # 红色

        # 更新视图
        self.vis_o3d.update_geometry(self.tree_lines)
        self.vis_o3d.update_geometry(self.path_lines)
        self.vis_o3d.poll_events()
        self.vis_o3d.update_renderer()
        time.sleep(0.01)
