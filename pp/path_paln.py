import trimesh
import networkx as nx
import numpy as np
import heapq
from scipy.spatial import KDTree

model_path = r"C:\Users\mi\Downloads\Three-Dimension-3D\models\pc\0\terra_obj\Block\Block.obj"

# Load the scene
scene = trimesh.load_scene(model_path)
mesh = list(scene.geometry.values())[0]
print(f"Mesh contains {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")

# Create a graph representation
G = nx.Graph()

# Add vertices to the graph
for i, vertex in enumerate(mesh.vertices):
    G.add_node(i, position=vertex)

# Add edges from the mesh faces
edges = set()
for face in mesh.faces:
    edges.add((face[0], face[1]))
    edges.add((face[1], face[2]))
    edges.add((face[2], face[0]))

# Add all edges with weights
for v1, v2 in edges:
    weight = np.linalg.norm(mesh.vertices[v1] - mesh.vertices[v2])
    G.add_edge(v1, v2, weight=weight)

print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Define waypoints
waypoints = [
    [78.56310916, 29.99410818, -162.10114156],  # start
    [83.32050134, 42.84368528, -164.97868412],
    [94.56436536, 52.84385806, -162.24338607],
    [77.65503504, 54.38683771, -166.85624655]  # end
]

# Create a KDTree for efficient nearest neighbor search
vertex_kdtree = KDTree(mesh.vertices)

# Add waypoints to the graph
waypoint_indices = []
for i, point in enumerate(waypoints):
    waypoint_idx = len(mesh.vertices) + i
    waypoint_indices.append(waypoint_idx)

    # Add waypoint as a node
    G.add_node(waypoint_idx, position=np.array(point))

    # Connect to nearest vertices (increase k for better connectivity)
    distances, indices = vertex_kdtree.query(point, k=50)  # Increased from 20 to 50

    # Connect waypoint to these vertices
    for j, idx in enumerate(indices):
        G.add_edge(waypoint_idx, idx, weight=distances[j])

# Identify connected components
components = list(nx.connected_components(G))
print(f"Graph has {len(components)} connected components")

# Map each waypoint to its component
waypoint_components = {}
for i, idx in enumerate(waypoint_indices):
    for j, component in enumerate(components):
        if idx in component:
            print(f"Waypoint {i} is in component {j} with {len(component)} nodes")
            waypoint_components[i] = j
            break

# SOLUTION: Connect the components that contain waypoints
# This creates "bridge" edges between disconnected parts of the mesh
waypoint_component_indices = {}
for i, comp_idx in waypoint_components.items():
    waypoint_component_indices[comp_idx] = i

# Get a representative vertex from each component containing a waypoint
component_representatives = {}
for comp_idx in waypoint_component_indices.keys():
    # Get a mesh vertex (not a waypoint) from this component
    for vertex in components[comp_idx]:
        if vertex < len(mesh.vertices):  # Ensure it's a mesh vertex
            component_representatives[comp_idx] = vertex
            break

# Connect the components with additional edges
for i in range(len(waypoint_components) - 1):
    comp1 = waypoint_components[i]
    comp2 = waypoint_components[i + 1]

    if comp1 != comp2:  # Only if they're different components
        v1 = waypoint_indices[i]  # Use the waypoint itself for more direct paths
        v2 = waypoint_indices[i + 1]

        # Add a direct edge between the waypoints
        weight = np.linalg.norm(
            np.array(G.nodes[v1]['position']) -
            np.array(G.nodes[v2]['position'])
        )
        G.add_edge(v1, v2, weight=weight)
        print(f"Added bridge edge from waypoint {i} to waypoint {i + 1}")

# Verify connectivity after adding bridge edges
if nx.is_connected(G):
    print("Graph is now connected! Path finding should succeed.")
else:
    print("Graph is still not fully connected. Additional bridges may be needed.")

# Check direct connectivity between waypoints
for i in range(len(waypoint_indices) - 1):
    start = waypoint_indices[i]
    end = waypoint_indices[i + 1]
    if nx.has_path(G, start, end):
        print(f"Waypoints {i} and {i + 1} are now connected in the graph")
    else:
        print(f"WARNING: Waypoints {i} and {i + 1} are still NOT connected in the graph")


# A* algorithm for pathfinding (unchanged from previous code)
def astar_path(graph, start_idx, goal_idx):
    if start_idx == goal_idx:
        return [start_idx]

    g_scores = {start_idx: 0}
    f_scores = {}
    came_from = {}
    open_set = []
    counter = 0

    start_pos = np.array(graph.nodes[start_idx]['position'])
    goal_pos = np.array(graph.nodes[goal_idx]['position'])

    h = np.linalg.norm(start_pos - goal_pos)
    f_scores[start_idx] = h

    heapq.heappush(open_set, (h, counter, start_idx))
    counter += 1

    open_set_hash = {start_idx}

    while open_set:
        _, _, current = heapq.heappop(open_set)
        open_set_hash.remove(current)

        if current == goal_idx:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_idx)
            return path[::-1]

        for neighbor in graph.neighbors(current):
            tentative_g = g_scores[current] + graph[current][neighbor]['weight']

            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                came_from[neighbor] = current
                g_scores[neighbor] = tentative_g

                neighbor_pos = np.array(graph.nodes[neighbor]['position'])
                h = np.linalg.norm(neighbor_pos - goal_pos)
                f_scores[neighbor] = tentative_g + h

                if neighbor not in open_set_hash:
                    counter += 1
                    heapq.heappush(open_set, (f_scores[neighbor], counter, neighbor))
                    open_set_hash.add(neighbor)

    return None


# Find paths between all consecutive waypoints
all_paths = []
for i in range(len(waypoint_indices) - 1):
    start_idx = waypoint_indices[i]
    end_idx = waypoint_indices[i + 1]

    print(f"Finding path from waypoint {i} to waypoint {i + 1}...")
    path = astar_path(G, start_idx, end_idx)

    if path:
        print(f"Path found with {len(path)} points")
        all_paths.append(path)
    else:
        print(f"Failed to find path between waypoints {i} and {i + 1}")

# Combine all paths
if len(all_paths) == len(waypoint_indices) - 1:
    # Combine paths, removing duplicates at segment joins
    full_path = all_paths[0]
    for path in all_paths[1:]:
        full_path.extend(path[1:])  # Skip first point as it's the same as last point of previous path

    # Extract coordinates
    path_coords = [G.nodes[idx]['position'] for idx in full_path]

    # Create a new scene
    path_scene = trimesh.Scene()

    # Add the original mesh with proper rendering settings
    mesh_copy = mesh.copy()
    # Set the mesh to display with proper lighting
    mesh_copy.visual.face_colors = [200, 200, 200, 255]  # Light gray
    path_scene.add_geometry(mesh_copy)

    # Add path segments as individual cylinders
    for i in range(len(path_coords) - 1):
        start = path_coords[i]
        end = path_coords[i + 1]

        # Create a cylinder for the path segment
        # Compute direction vector and length
        direction = end - start
        length = np.linalg.norm(direction)

        if length > 0:  # Skip zero-length segments
            # Create a cylinder
            cylinder = trimesh.creation.cylinder(
                radius=0.3,  # Adjust for visibility
                segment=[start, end],
                sections=8  # Lower for better performance
            )
            cylinder.visual.face_colors = [255, 0, 0, 255]  # Red for path
            path_scene.add_geometry(cylinder)

    # Add waypoint markers
    for i, point in enumerate(waypoints):
        # Create a sphere for each waypoint - without the sections parameter
        sphere = trimesh.primitives.Sphere(
            center=point,
            radius=0.5
        )

        # Different colors for start, intermediate, and end points
        if i == 0:
            sphere.visual.face_colors = [0, 255, 0, 255]  # Green for start
        elif i == len(waypoints) - 1:
            sphere.visual.face_colors = [255, 0, 0, 255]  # Red for end
        else:
            sphere.visual.face_colors = [0, 0, 255, 255]  # Blue for intermediate

        path_scene.add_geometry(sphere)

    # Save the visualization to files (for reliable viewing)
    path_scene.export('path_visualization.glb')  # GLB format for 3D viewers

    # Save a PNG render of the scene
    png = path_scene.save_image(resolution=(1024, 768))
    with open('path_visualization.png', 'wb') as f:
        f.write(png)

    print(f"Path visualization exported to path_visualization.glb and path_visualization.png")

    # Show the scene
    try:
        path_scene.show()
    except Exception as e:
        print(f"Error displaying scene: {e}")
        print("Please check the exported files instead.")
else:
    print("Could not find complete path through all waypoints.")

    # Diagnostic visualization
    diag_scene = trimesh.Scene()
    diag_scene.add_geometry(mesh)

    # Add spheres for waypoints
    for i, point in enumerate(waypoints):
        sphere = trimesh.primitives.Sphere(center=point, radius=0.5)
        sphere.visual.face_colors = [255, 0, 0, 255]
        diag_scene.add_geometry(sphere)

    diag_scene.export('diagnostic.glb')
    diag_scene.show()