"""
Condor Buildings Generator - Mesh Converter

Converts MeshData instances from the pipeline to Blender mesh objects.
Handles vertex conversion, face indexing, and UV coordinate mapping.
"""

import bpy
from typing import List, Optional

# Import MeshData type for type hints (conditional to allow testing outside Blender)
try:
    from ..models.mesh import MeshData
except ImportError:
    MeshData = None


def meshdata_to_blender(
    mesh_data,  # MeshData
    name: str = "building",
    collection: Optional[bpy.types.Collection] = None
) -> bpy.types.Object:
    """
    Convert a MeshData instance to a Blender mesh object.

    Args:
        mesh_data: Pipeline MeshData with vertices, faces, and UVs
        name: Base name for the object (osm_id will be used if available)
        collection: Target collection (defaults to active collection)

    Returns:
        Created Blender object

    Note:
        - MeshData uses 1-based indices (OBJ convention)
        - Blender uses 0-based indices
        - Conversion is handled automatically
    """
    # Use osm_id as name if available
    if mesh_data.osm_id:
        name = f"building_{mesh_data.osm_id}"

    # Create new mesh and object
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)

    # Convert vertices: MeshData stores as tuples (x, y, z)
    vertices = list(mesh_data.vertices)

    # Convert faces: MeshData uses 1-based indices, Blender uses 0-based
    faces = [[idx - 1 for idx in face] for face in mesh_data.faces]

    # Create mesh geometry
    # Note: from_pydata expects vertices, edges, faces
    mesh.from_pydata(vertices, [], faces)

    # Add UV layer if UVs exist
    if mesh_data.uvs and mesh_data.face_uvs:
        _add_uv_layer(mesh, mesh_data)

    # Update mesh to compute normals, etc.
    mesh.update()

    # Link object to collection
    if collection is None:
        collection = bpy.context.collection
    collection.objects.link(obj)

    return obj


def _add_uv_layer(mesh: bpy.types.Mesh, mesh_data) -> None:
    """
    Add UV coordinates to a Blender mesh.

    Blender stores UVs per-loop (per corner of each face), not per-vertex.
    This function maps MeshData's face_uvs to Blender's loop-based UV system.

    Args:
        mesh: Blender mesh to add UVs to
        mesh_data: MeshData with uvs and face_uvs
    """
    # Create UV layer
    uv_layer = mesh.uv_layers.new(name="UVMap")

    # Blender's UV data is accessed per-loop
    # mesh.polygons[i].loop_indices gives the loop indices for face i
    for face_idx, polygon in enumerate(mesh.polygons):
        # Get UV indices for this face (1-based from MeshData)
        if face_idx < len(mesh_data.face_uvs):
            face_uv_indices = mesh_data.face_uvs[face_idx]

            # Map each loop to its UV
            for loop_local_idx, loop_idx in enumerate(polygon.loop_indices):
                if loop_local_idx < len(face_uv_indices):
                    # Convert 1-based UV index to 0-based
                    uv_idx = face_uv_indices[loop_local_idx] - 1

                    if 0 <= uv_idx < len(mesh_data.uvs):
                        uv = mesh_data.uvs[uv_idx]
                        uv_layer.data[loop_idx].uv = (uv[0], uv[1])


def create_buildings_collection(
    name: str = "Condor Buildings",
    parent: Optional[bpy.types.Collection] = None
) -> bpy.types.Collection:
    """
    Create or get a collection for imported buildings.

    Args:
        name: Collection name
        parent: Parent collection (defaults to scene collection)

    Returns:
        The collection (created or existing)
    """
    # Check if collection already exists
    if name in bpy.data.collections:
        return bpy.data.collections[name]

    # Create new collection
    collection = bpy.data.collections.new(name)

    # Link to parent
    if parent is None:
        parent = bpy.context.scene.collection
    parent.children.link(collection)

    return collection


def import_meshes_to_blender(
    meshes: List,  # List[MeshData]
    collection_name: str = "Condor Buildings",
    join_meshes: bool = False
) -> List[bpy.types.Object]:
    """
    Import multiple MeshData instances to Blender.

    Args:
        meshes: List of MeshData from pipeline
        collection_name: Name for the collection to hold buildings
        join_meshes: If True, join all meshes into single object (faster for large datasets)

    Returns:
        List of created Blender objects
    """
    if not meshes:
        return []

    # Create collection for buildings
    collection = create_buildings_collection(collection_name)

    # Import each mesh
    objects = []
    for mesh_data in meshes:
        if mesh_data.vertices:  # Skip empty meshes
            obj = meshdata_to_blender(mesh_data, collection=collection)
            objects.append(obj)

    # Optionally join all objects for performance
    if join_meshes and len(objects) > 1:
        # Select all objects
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects:
            obj.select_set(True)

        # Set active object
        bpy.context.view_layer.objects.active = objects[0]

        # Join
        bpy.ops.object.join()

        # Return single joined object
        return [bpy.context.active_object]

    return objects


def cleanup_buildings_collection(name: str = "Condor Buildings") -> int:
    """
    Remove all objects from a buildings collection.

    Args:
        name: Collection name to clean up

    Returns:
        Number of objects removed
    """
    if name not in bpy.data.collections:
        return 0

    collection = bpy.data.collections[name]
    count = len(collection.objects)

    # Remove all objects
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    return count
