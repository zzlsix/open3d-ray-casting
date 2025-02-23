from dataclasses import dataclass

import numpy as np
import open3d as o3d


def ray_triangle_intersect(ray_origin, ray_direction, v0, v1, v2) -> tuple[bool, np.ndarray | None]:
    epsilon = 1e-8
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    if -epsilon < a < epsilon:
        return False, None  # This ray is parallel to this triangle.
    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)
    if not (0.0 <= u <= 1.0):
        return False, None
    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    if not (0.0 <= v <= 1.0):
        return False, None
    if u + v > 1.0:
        return False, None
    t = f * np.dot(edge2, q)
    if t > epsilon:
        intersect_point = ray_origin + ray_direction * t
        return True, intersect_point
    else:
        return (
            False,
            None,
        )  # This means that there is a line intersection but not a ray intersection.


class ViewStateManager:
    def __init__(self) -> None:
        self.prior_mouse_position: tuple[float, float] | None = None
        self.is_view_rotating = False
        self.is_translating = False
        self.pixel_to_rotate_scale_factor = 1
        self.pixel_to_translate_scale_factor = 1
        self.clickable_geometries: dict[str, o3d.geometry.TriangleMesh] = {}
        self.selected_mesh: str | None = None

    def add_clickable_geometry(self, id: str, geometry: o3d.geometry.TriangleMesh):
        self.clickable_geometries[id] = geometry

    def on_mouse_move(self, vis, x, y):
        if self.prior_mouse_position is not None:
            dx = x - self.prior_mouse_position[0]
            dy = y - self.prior_mouse_position[1]
            view_control = vis.get_view_control()
            if self.is_view_rotating:
                view_control.rotate(
                    dx * self.pixel_to_rotate_scale_factor, dy * self.pixel_to_rotate_scale_factor
                )
            elif self.is_translating:
                view_control.translate(
                    dx * self.pixel_to_translate_scale_factor,
                    dy * self.pixel_to_translate_scale_factor,
                )

        self.prior_mouse_position = (x, y)

    def on_mouse_scroll(self, vis, x, y):
        view_control = vis.get_view_control()
        view_control.scale(y)

    def on_mouse_button(self, vis, button, action, mods):
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
            self.pick_mesh(vis, self.prior_mouse_position[0], self.prior_mouse_position[1])

        print(f"on_mouse_button: {button}, {action}, {mods}")
        # print(vis.get_view_status())
        if button == "right" and action == "down":
            self.pick_mesh(vis, self.prior_mouse_position[0], self.prior_mouse_position[1])

    def pick_mesh(self, vis, x, y):
        view_control = vis.get_view_control()
        camera_params = view_control.convert_to_pinhole_camera_parameters()
        intrinsic = camera_params.intrinsic.intrinsic_matrix
        extrinsic = camera_params.extrinsic

        # Create a ray in camera space
        ray_camera = np.array(
            [
                (x - intrinsic[0, 2]) / intrinsic[0, 0],
                (y - intrinsic[1, 2]) / intrinsic[1, 1],
                1.0,
            ]
        )

        # Normalize the ray direction
        ray_camera = ray_camera / np.linalg.norm(ray_camera)

        # Convert the ray to world space
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]

        ray_world = np.dot(rotation.T, ray_camera)
        ray_dir = ray_world / np.linalg.norm(ray_world)

        camera_pos = -np.dot(rotation.T, translation)

        # Add sphere at camera position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(camera_pos)
        vis.add_geometry(sphere, reset_bounding_box=False)

        # Draw the ray in world space
        ray_end = camera_pos + ray_dir * 100  # Extend the ray 100 units
        ray_line = o3d.geometry.LineSet()
        ray_line.points = o3d.utility.Vector3dVector([camera_pos, ray_end])
        ray_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        ray_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
        vis.add_geometry(ray_line, reset_bounding_box=False)

        closest_mesh_lookup: dict[str, float] = {}
        for id, mesh in self.clickable_geometries.items():
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            for tri in triangles:
                v0, v1, v2 = vertices[tri]
                hit, intersect_point = ray_triangle_intersect(camera_pos, ray_dir, v0, v1, v2)
                if hit:
                    intersection_distance = np.linalg.norm(intersect_point - camera_pos)
                    closest_mesh_lookup[id] = min(
                        intersection_distance, closest_mesh_lookup.get(id, np.inf)
                    )

        if len(closest_mesh_lookup) == 0:
            print("No hit detected")
            return

        closest_mesh_id = min(closest_mesh_lookup, key=closest_mesh_lookup.get)
        print(f"Closest mesh: {closest_mesh_id}")
        self.selected_mesh = closest_mesh_id


def custom_mouse_action(pcd):
    vis = o3d.visualization.VisualizerWithKeyCallback()

    state_manager = ViewStateManager()
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.1, 0.1, 0.7])
    mesh_box.compute_vertex_normals()
    mesh_box.compute_triangle_normals()
    mesh_box.translate([0.0, 0.0, 0.0])

    mesh_box2 = o3d.geometry.TriangleMesh.create_box(width=1.0, height=0.8, depth=1.0)
    mesh_box2.compute_vertex_normals()
    mesh_box2.paint_uniform_color([0.7, 0.1, 0.1])
    mesh_box2.compute_vertex_normals()
    mesh_box2.compute_triangle_normals()
    mesh_box2.translate([0.0, 0.0, 2.0])

    state_manager.add_clickable_geometry("mesh_box", mesh_box)
    state_manager.add_clickable_geometry("mesh_box2", mesh_box2)

    vis.register_mouse_move_callback(state_manager.on_mouse_move)
    vis.register_mouse_scroll_callback(state_manager.on_mouse_scroll)
    vis.register_mouse_button_callback(state_manager.on_mouse_button)

    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(mesh_box)
    vis.add_geometry(mesh_box2)
    vis.run()


if __name__ == "__main__":
    ply_data = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(ply_data.path)

    print("Customized visualization with mouse action.")
    custom_mouse_action(pcd)