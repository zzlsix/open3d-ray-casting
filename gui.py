import open3d as o3d
import numpy as np
from typing import Tuple, List, Optional
from open3d.cpu.pybind.camera import PinholeCameraParameters, PinholeCameraIntrinsic
from open3d.cpu.pybind.visualization import ViewControl, VisualizerWithKeyCallback
from open3d.visualization import Visualizer


# AIGC START
class ModelWindow:
    def __init__(self, width: int, height: int, title: str):
        self.window = o3d.visualization.gui.Application.instance.create_window(
            title, width, height)
        self.widget3d = o3d.visualization.gui.SceneWidget()
        self.window.add_child(self.widget3d)

        # 设置widget3d填充整个窗口
        self.window.set_on_layout(self._on_layout)
        self.widget3d.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)

    def _on_layout(self, layout_context):
        content_rect = self.window.content_rect
        self.widget3d.frame = content_rect


class ModelInteractor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.clicked_points: List[Tuple[float, float]] = []
        self.last_cursor_pos: Optional[Tuple[float, float]] = None
        self.mesh = None
        self.scene = None
        self.window = None
        self.widget3d = None

    def load_model(self) -> None:
        """加载3D模型并初始化场景"""
        # 加载模型时保留纹理信息
        self.mesh = o3d.io.read_triangle_mesh(self.model_path,
                                              enable_post_processing=True,
                                              print_progress=True)
        self.mesh.compute_vertex_normals()

        # 初始化射线投射场景
        self.scene = o3d.t.geometry.RaycastingScene()

        # 处理多材质模型
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)

        # 创建tensor mesh并添加到场景
        mesh_t = o3d.t.geometry.TriangleMesh()
        mesh_t.vertex.positions = o3d.core.Tensor(vertices.astype(np.float32))
        mesh_t.triangle.indices = o3d.core.Tensor(triangles.astype(np.int32))

        # 打印一些调试信息
        print(f"Tensor mesh created:")
        print(f"Vertices shape: {mesh_t.vertex.positions.shape}")
        print(f"Triangles shape: {mesh_t.triangle.indices.shape}")

        # 添加到场景
        scene_id = self.scene.add_triangles(mesh_t)
        print(f"Added to scene with ID: {scene_id}")

    def initialize_window(self, width: int = 1024, height: int = 768):
        """初始化GUI窗口"""
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        self.window = ModelWindow(width, height, "3D模型交互器")
        self.widget3d = self.window.widget3d

        # 使用模型原有的材质
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"

        # 直接添加模型，保留原有纹理
        self.widget3d.scene.add_geometry("model", self.mesh, material)

        # 设置默认相机视角
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60, bounds, bounds.get_center())

        # 打印场景信息
        print(f"Scene bounding box: {bounds}")  # 调试信息
        print(f"Model vertices: {len(self.mesh.vertices)}")  # 调试信息
        print(f"Model triangles: {len(self.mesh.triangles)}")  # 调试信息

        # 注册事件回调
        self.widget3d.set_on_mouse(self._handle_mouse_event)

        # 显示窗口
        app.run()

    def _handle_mouse_event(self, event):
        """处理鼠标事件"""
        # 更新鼠标位置
        self.last_cursor_pos = (event.x, event.y)

        # 处理鼠标点击
        if event.type == o3d.visualization.gui.MouseEvent.Type.BUTTON_DOWN:
            print(f"Mouse button down at ({event.x}, {event.y})")  # 调试信息

            if event.is_modifier_down(o3d.visualization.gui.KeyModifier.SHIFT):
                print("Shift is pressed")  # 调试信息

                if event.is_button_down(o3d.visualization.gui.MouseButton.LEFT):
                    print("Left button is pressed")  # 调试信息

                    # 获取相机参数
                    camera = self.widget3d.scene.camera

                    # 获取视图矩阵和投影矩阵
                    view_matrix = np.array(camera.get_view_matrix())
                    projection_matrix = np.array(camera.get_projection_matrix())
                    print(f"View matrix:\n{view_matrix}")  # 调试信息
                    print(f"Projection matrix:\n{projection_matrix}")  # 调试信息

                    # 获取视口大小
                    frame = self.widget3d.frame
                    width = frame.width
                    height = frame.height
                    print(f"Window size: {width}x{height}")  # 调试信息

                    if width == 0 or height == 0:
                        print("Invalid window size")  # 调试信息
                        return True

                    # 计算归一化设备坐标 (NDC)
                    x = (2.0 * event.x) / width - 1.0
                    y = 1.0 - (2.0 * event.y) / height
                    print(f"NDC coordinates: ({x}, {y})")  # 新增调试信息

                    # 获取相机位置（射线原点）
                    camera_pos = np.array([
                        -view_matrix[0, 3],
                        -view_matrix[1, 3],
                        -view_matrix[2, 3]
                    ])
                    print(f"Camera position: {camera_pos}")

                    # 计算射线方向
                    # 使用NDC坐标和投影矩阵计算视图空间中的方向
                    view_dir = np.array([x / projection_matrix[0, 0],
                                         y / projection_matrix[1, 1],
                                         -1.0])  # 指向-z方向
                    print(f"View space direction: {view_dir}")

                    # 将方向转换到世界空间
                    rotation = view_matrix[:3, :3]
                    world_dir = np.dot(np.linalg.inv(rotation), view_dir)
                    world_dir = world_dir / np.linalg.norm(world_dir)
                    print(f"World space direction: {world_dir}")

                    # 构建射线
                    ray_origin = camera_pos
                    ray_direction = world_dir

                    print(f"Ray origin: {ray_origin}")
                    print(f"Ray direction: {ray_direction}")

                    # 计算射线与模型的交点
                    rays = o3d.core.Tensor([[*ray_origin, *ray_direction]],
                                           dtype=o3d.core.Dtype.Float32)
                    print(f"Ray tensor created: {rays.numpy()}")

                    intersection_result = self.scene.cast_rays(rays)
                    print(f"Intersection result: {intersection_result}")

                    t_hit = intersection_result['t_hit'].numpy()
                    print(f"Hit distance: {t_hit[0]}")

                    if t_hit[0] >= 0 and t_hit[0] != float('inf'):  # 检查是否有有效的交点
                        intersection_point = ray_origin + ray_direction * t_hit[0]
                        if not np.any(np.isnan(intersection_point)) and not np.any(np.isinf(intersection_point)):
                            # 获取更多交点信息
                            primitive_normal = intersection_result['primitive_normals'].numpy()[0]
                            primitive_uv = intersection_result['primitive_uvs'].numpy()[0]
                            primitive_id = intersection_result['primitive_ids'].numpy()[0]

                            # 检查法线是否有效
                            if not np.any(np.isnan(primitive_normal)) and not np.all(primitive_normal == 0):
                                print(f"击中三角形ID: {primitive_id}")
                                print(f"表面法线: {primitive_normal}")
                                print(f"UV坐标: {primitive_uv}")
                                print(f"交点坐标: {intersection_point}")

                                # 保存点击位置
                                self.clicked_points.append((event.x, event.y))

                                # 根据法线调整标记球的大小
                                normal_z = np.abs(primitive_normal[2])  # 法线的z分量
                                sphere_radius = 0.01 * (0.5 + 0.5 * normal_z)  # 根据法线调整大小

                                try:
                                    # 创建标记球
                                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
                                    sphere.compute_vertex_normals()
                                    sphere.translate(intersection_point)

                                    # 根据法线设置标记颜色
                                    color = [0.8 + 0.2 * primitive_normal[0],  # R
                                             0.8 + 0.2 * primitive_normal[1],  # G
                                             0.8 + 0.2 * primitive_normal[2]]  # B

                                    # 确保颜色值在有效范围内
                                    color = [max(0.0, min(1.0, c)) for c in color]
                                    sphere.paint_uniform_color(color)

                                    # 添加到场景
                                    marker_material = o3d.visualization.rendering.MaterialRecord()
                                    marker_material.shader = "defaultLit"
                                    marker_material.base_color = color + [1.0]  # 添加alpha通道

                                    marker_name = f"marker_{len(self.clicked_points)}"
                                    self.widget3d.scene.add_geometry(
                                        marker_name,
                                        sphere,
                                        marker_material
                                    )
                                    print(f"成功添加标记: {marker_name}")
                                except Exception as e:
                                    print(f"添加标记失败: {str(e)}")
                            else:
                                print("无效的表面法线")
                        else:
                            print("无效的交点坐标")
                    else:
                        print("射线未击中模型")

        return True


def main():
    MODEL_PATH = r"C:\Users\mi\Downloads\Three-Dimension-3D\models\pc\0\terra_obj\Block\Block.obj"

    # 初始化交互器
    interactor = ModelInteractor(MODEL_PATH)
    interactor.load_model()
    interactor.initialize_window()


if __name__ == "__main__":
    main()
# AIGC END
