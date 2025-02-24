import open3d as o3d
import numpy as np
from typing import Tuple, List, Optional
from open3d.cpu.pybind.camera import PinholeCameraParameters, PinholeCameraIntrinsic
from open3d.cpu.pybind.visualization import ViewControl, VisualizerWithKeyCallback

class ModelInteractor:
    def __init__(self, model_path: str):
        # 模型路径
        self.model_path = model_path
        # 保存点击的点
        self.clicked_points: List[Tuple[float, float]] = []
        # 鼠标位移的位置
        self.last_cursor_pos: Optional[Tuple[float, float]] = None
        # 标记旋转
        self.is_view_rotating = False
        # 标记翻转
        self.is_translating = False
        self.pixel_to_rotate_scale_factor = 1
        self.pixel_to_translate_scale_factor = 1
        self.mesh = None
        self.scene = None

    def load_model(self) -> None:
        """加载3D模型并初始化场景"""
        self.mesh = o3d.io.read_triangle_mesh(self.model_path, enable_post_processing=True)
        self.mesh.compute_vertex_normals()
        # 初始化射线投射场景
        self.scene = o3d.t.geometry.RaycastingScene()
        temp_mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        self.scene.add_triangles(temp_mesh)

    def screen_to_world_ray(self, 
                          cursor_pos: Tuple[float, float], 
                          view_control: ViewControl) -> Tuple[np.ndarray, np.ndarray]:
        """将屏幕坐标转换为世界坐标系中的射线"""
        u, v = cursor_pos
        camera_params: PinholeCameraParameters = view_control.convert_to_pinhole_camera_parameters()
        intrinsic: PinholeCameraIntrinsic = camera_params.intrinsic
        extrinsic = camera_params.extrinsic

        # 计算NDC坐标
        ndc_x = (2.0 * u / intrinsic.width) - 1.0
        ndc_y = 1.0 - (2.0 * v / intrinsic.height)

        # 获取相机参数
        fx, fy = intrinsic.get_focal_length()
        cx, cy = intrinsic.get_principal_point()

        # 计算像素坐标
        pixel_x = ((ndc_x + 1.0) / 2.0) * intrinsic.width
        pixel_y = ((1.0 - ndc_y) / 2.0) * intrinsic.height

        # 计算相机坐标系中的射线方向
        ray_dir = np.array([(pixel_x - cx) / fx, 
                           (pixel_y - cy) / fy, 
                           1.0])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        # 转换到世界坐标系
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
        ray_origin = -np.dot(rotation.T, translation)
        world_ray_dir = np.dot(rotation.T, ray_dir)
        world_ray_dir = world_ray_dir / np.linalg.norm(world_ray_dir)

        return ray_origin, world_ray_dir

    def ray_mesh_intersection(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> Optional[np.ndarray]:
        """计算射线与模型的交点"""
        rays = o3d.core.Tensor([[*ray_origin, *ray_direction]], dtype=o3d.core.Dtype.Float32)
        intersection_result = self.scene.cast_rays(rays)
        
        t_hit = intersection_result['t_hit'].numpy()
        if t_hit[0] >= 0:
            intersection_point = ray_origin + ray_direction * t_hit[0]
            return intersection_point
        return None

    def handle_mouse_click(self, vis: VisualizerWithKeyCallback, button: int, action: int, mods: int) -> bool:
        """处理鼠标点击事件"""
        buttons = ["left", "right", "middle"]
        actions = ["up", "down"]
        mods_name = ["shift", "ctrl", "alt", "cmd"]

        button = buttons[button]
        action = actions[action]
        mods = [mods_name[i] for i in range(4) if mods & (1 << i)]

        if button == "left" and action == "down":
            self.is_view_rotating = True
        elif button == "left" and action == "up":
            self.is_view_rotating = False
        elif button == "middle" and action == "down":
            self.is_translating = True
        elif button == "middle" and action == "up":
            self.is_translating = False
        elif button == "right" and action == "down":
            ray_origin, ray_dir = self.screen_to_world_ray(
                self.last_cursor_pos,
                vis.get_view_control()
            )
            intersection_point = self.ray_mesh_intersection(ray_origin, ray_dir)
            if intersection_point is not None:
                print(f"交点坐标: {intersection_point}")
                self.clicked_points.append(self.last_cursor_pos)
        return False

    def handle_mouse_move(self, vis: VisualizerWithKeyCallback, x: float, y: float) -> bool:
        """处理鼠标移动事件"""
        if self.last_cursor_pos is not None:
            move_x = x - self.last_cursor_pos[0]
            move_y = y - self.last_cursor_pos[1]
            view_control: ViewControl = vis.get_view_control()
            if self.is_view_rotating:
                view_control.rotate(move_x * self.pixel_to_rotate_scale_factor, move_y * self.pixel_to_rotate_scale_factor)
            elif self.is_translating:
                view_control.translate(move_x * self.pixel_to_translate_scale_factor, move_y * self.pixel_to_translate_scale_factor)
        self.last_cursor_pos = (x, y)
        return False

def main():
    MODEL_PATH = r"C:\Users\mi\Downloads\Three-Dimension-3D\models\pc\0\terra_obj\Block\Block.obj"
    
    # 初始化交互器
    interactor = ModelInteractor(MODEL_PATH)
    interactor.load_model()

    # 创建可视化窗口
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="3D模型交互器")
    vis.add_geometry(interactor.mesh)

    # 注册回调函数
    vis.register_mouse_move_callback(interactor.handle_mouse_move)
    vis.register_mouse_button_callback(interactor.handle_mouse_click)

    # 运行可视化
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
