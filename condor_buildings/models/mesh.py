"""
Mesh data model for Condor Buildings Generator.

Provides MeshData class for representing generated building geometry
that can be merged and exported to OBJ format.

Phase B preparation: UV coordinate support for texture mapping.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class MeshData:
    """
    Generated mesh data for buildings.

    Stores vertices, UVs, and faces that can be merged and exported.
    Faces use 1-based indexing for OBJ compatibility.

    Attributes:
        vertices: List of (x, y, z) vertex positions
        uvs: List of (u, v) texture coordinates (Phase B)
        faces: List of face vertex indices (1-indexed for OBJ)
        face_uvs: List of face UV indices (1-indexed, parallel to faces)
        osm_id: Optional building ID for grouping in export

    UV Strategy (Phase B):
        - Walls use a "strip atlas": tileable in U, variants stacked in V
        - U coordinate can exceed [0,1] for wrapping without atlas jumps
        - V coordinate selects texture variant in the atlas

    Note on indexing:
        - Internal faces are stored with 1-based indices
        - When merging, indices are adjusted by vertex/UV offset
    """
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    uvs: List[Tuple[float, float]] = field(default_factory=list)
    faces: List[List[int]] = field(default_factory=list)
    face_uvs: List[List[int]] = field(default_factory=list)  # Parallel to faces
    osm_id: Optional[str] = None

    def vertex_count(self) -> int:
        """Get number of vertices."""
        return len(self.vertices)

    def uv_count(self) -> int:
        """Get number of UV coordinates."""
        return len(self.uvs)

    def face_count(self) -> int:
        """Get number of faces."""
        return len(self.faces)

    def triangle_count(self) -> int:
        """
        Get number of triangles.

        Assumes all faces are triangles. If faces are quads or n-gons,
        this may not be accurate.
        """
        return len(self.faces)

    def has_uvs(self) -> bool:
        """Check if mesh has UV coordinates."""
        return len(self.uvs) > 0

    def add_vertex(self, x: float, y: float, z: float) -> int:
        """
        Add a vertex and return its 1-based index.

        Args:
            x, y, z: Vertex coordinates

        Returns:
            1-based index of the new vertex
        """
        self.vertices.append((x, y, z))
        return len(self.vertices)  # 1-based

    def add_uv(self, u: float, v: float) -> int:
        """
        Add a UV coordinate and return its 1-based index.

        Args:
            u, v: Texture coordinates

        Returns:
            1-based index of the new UV
        """
        self.uvs.append((u, v))
        return len(self.uvs)  # 1-based

    def add_vertex_with_uv(
        self, x: float, y: float, z: float, u: float, v: float
    ) -> Tuple[int, int]:
        """
        Add a vertex and UV coordinate together.

        Args:
            x, y, z: Vertex coordinates
            u, v: Texture coordinates

        Returns:
            Tuple of (vertex_index, uv_index), both 1-based
        """
        v_idx = self.add_vertex(x, y, z)
        uv_idx = self.add_uv(u, v)
        return (v_idx, uv_idx)

    def add_face(self, indices: List[int]) -> None:
        """
        Add a face with given vertex indices (1-based).

        Args:
            indices: List of vertex indices (1-based)
        """
        self.faces.append(indices)

    def add_face_with_uvs(
        self, vertex_indices: List[int], uv_indices: List[int]
    ) -> None:
        """
        Add a face with vertex and UV indices.

        Args:
            vertex_indices: List of vertex indices (1-based)
            uv_indices: List of UV indices (1-based), same length as vertex_indices
        """
        self.faces.append(vertex_indices)
        self.face_uvs.append(uv_indices)

    def add_triangle(self, v1: int, v2: int, v3: int) -> None:
        """
        Add a triangle face.

        Args:
            v1, v2, v3: Vertex indices (1-based)
        """
        self.faces.append([v1, v2, v3])

    def add_triangle_with_uvs(
        self,
        v1: int, v2: int, v3: int,
        uv1: int, uv2: int, uv3: int
    ) -> None:
        """
        Add a triangle face with UVs.

        Args:
            v1, v2, v3: Vertex indices (1-based)
            uv1, uv2, uv3: UV indices (1-based)
        """
        self.faces.append([v1, v2, v3])
        self.face_uvs.append([uv1, uv2, uv3])

    def add_quad(self, v1: int, v2: int, v3: int, v4: int) -> None:
        """
        Add a quad face (will be triangulated).

        Splits quad into two triangles: (v1, v2, v3) and (v1, v3, v4)

        Args:
            v1, v2, v3, v4: Vertex indices (1-based, CCW order)
        """
        self.faces.append([v1, v2, v3])
        self.faces.append([v1, v3, v4])

    def add_quad_with_uvs(
        self,
        v1: int, v2: int, v3: int, v4: int,
        uv1: int, uv2: int, uv3: int, uv4: int
    ) -> None:
        """
        Add a quad face with UVs (will be triangulated).

        Splits quad into two triangles with corresponding UVs.

        Args:
            v1, v2, v3, v4: Vertex indices (1-based, CCW order)
            uv1, uv2, uv3, uv4: UV indices (1-based)
        """
        self.faces.append([v1, v2, v3])
        self.face_uvs.append([uv1, uv2, uv3])
        self.faces.append([v1, v3, v4])
        self.face_uvs.append([uv1, uv3, uv4])

    def add_polygon(self, *vertices: int) -> None:
        """
        Add an n-gon face (polygon with any number of vertices).

        Unlike add_quad, this keeps the polygon as a single face without
        triangulation. Useful for pentagons and other n-gons where we want
        to preserve the face structure for UV mapping.

        Args:
            *vertices: Vertex indices (1-based, CCW order for outward normal)
        """
        self.faces.append(list(vertices))

    def add_polygon_with_uvs(self, vertex_indices: List[int], uv_indices: List[int]) -> None:
        """
        Add an n-gon face with UVs.

        Args:
            vertex_indices: List of vertex indices (1-based, CCW order)
            uv_indices: List of UV indices (1-based), same length as vertex_indices
        """
        self.faces.append(vertex_indices)
        self.face_uvs.append(uv_indices)

    def merge(self, other: 'MeshData') -> None:
        """
        Merge another mesh into this one.

        Vertices, UVs, and faces are appended with indices adjusted.

        Args:
            other: MeshData to merge into this one
        """
        if not other.vertices:
            return

        vertex_offset = len(self.vertices)
        uv_offset = len(self.uvs)

        self.vertices.extend(other.vertices)
        self.uvs.extend(other.uvs)

        for face in other.faces:
            # Adjust indices by offset (faces are 1-based)
            new_face = [idx + vertex_offset for idx in face]
            self.faces.append(new_face)

        for face_uv in other.face_uvs:
            # Adjust UV indices by offset (1-based)
            new_face_uv = [idx + uv_offset for idx in face_uv]
            self.face_uvs.append(new_face_uv)

    def clear(self) -> None:
        """Clear all vertices, UVs, and faces."""
        self.vertices.clear()
        self.uvs.clear()
        self.faces.clear()
        self.face_uvs.clear()

    def is_empty(self) -> bool:
        """Check if mesh has no geometry."""
        return len(self.vertices) == 0

    def validate(self) -> List[str]:
        """
        Validate mesh integrity.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not self.vertices:
            errors.append("Mesh has no vertices")
            return errors

        max_idx = len(self.vertices)

        for i, face in enumerate(self.faces):
            if len(face) < 3:
                errors.append(f"Face {i} has fewer than 3 vertices")

            for idx in face:
                if idx < 1 or idx > max_idx:
                    errors.append(
                        f"Face {i} has invalid vertex index {idx} "
                        f"(valid range: 1-{max_idx})"
                    )

        return errors

    def compute_bounds(self) -> Optional[Tuple[Tuple[float, float, float],
                                                 Tuple[float, float, float]]]:
        """
        Compute bounding box of the mesh.

        Returns:
            ((min_x, min_y, min_z), (max_x, max_y, max_z)) or None if empty
        """
        if not self.vertices:
            return None

        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]

        return (
            (min(xs), min(ys), min(zs)),
            (max(xs), max(ys), max(zs))
        )

    def __repr__(self) -> str:
        return f"MeshData(vertices={len(self.vertices)}, faces={len(self.faces)})"


def create_empty_mesh(osm_id: Optional[str] = None) -> MeshData:
    """
    Create an empty mesh with optional OSM ID.

    Args:
        osm_id: Optional building ID for grouping

    Returns:
        Empty MeshData instance
    """
    return MeshData(osm_id=osm_id)


def merge_meshes(meshes: List[MeshData]) -> MeshData:
    """
    Merge multiple meshes into one.

    Args:
        meshes: List of MeshData to merge

    Returns:
        Single merged MeshData
    """
    result = MeshData()

    for mesh in meshes:
        result.merge(mesh)

    return result
