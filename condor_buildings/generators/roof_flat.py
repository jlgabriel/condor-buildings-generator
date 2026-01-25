"""
Flat roof generator for Condor Buildings Generator.

Generates flat roof surfaces with proper hole handling
using triangulation. Falls back gracefully on triangulation failure.

UV Mapping (Phase 2):
- Flat roofs use the same roof texture atlas as gabled roofs
- Each triangle gets UV coordinates from the roof pattern slice
- UVs are computed based on world-space XY position for seamless tiling
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

from ..models.geometry import Point2D, Polygon
from ..models.mesh import MeshData
from ..models.building import BuildingRecord
from ..utils.triangulation import (
    triangulate_polygon,
    triangulate_with_holes,
    TriangulationError,
)
from .uv_mapping import select_building_variations, get_roof_v_range, ROOF_SLICE_V

logger = logging.getLogger(__name__)


@dataclass
class FlatRoofConfig:
    """Configuration for flat roof generation."""
    pass  # Reserved for future options


def generate_flat_roof(
    building: BuildingRecord,
    config: Optional[FlatRoofConfig] = None
) -> MeshData:
    """
    Generate flat roof mesh for a building.

    Uses triangulation to handle complex footprints and holes.
    Falls back to outer ring only if triangulation with holes fails.

    UV mapping: Uses roof texture atlas with deterministic pattern selection.

    Args:
        building: Building record with footprint and heights
        config: Optional configuration

    Returns:
        MeshData with roof geometry
    """
    mesh = MeshData(osm_id=building.osm_id)

    roof_z = building.wall_top_z  # Flat roof at eave height
    footprint = building.footprint

    # Get texture variation for this building (deterministic)
    roof_index, _ = select_building_variations(building.seed)

    if footprint.has_holes:
        # Try triangulation with holes
        try:
            _generate_roof_with_holes(mesh, footprint, roof_z, roof_index)
        except TriangulationError as e:
            logger.warning(
                f"Building {building.osm_id}: Triangulation with holes failed: {e}. "
                f"Falling back to outer ring only."
            )
            building.warnings.append(f"Flat roof: holes ignored due to {e}")
            _generate_simple_roof(mesh, footprint.outer_ring, roof_z, roof_index)
    else:
        # Simple case: no holes
        try:
            _generate_simple_roof(mesh, footprint.outer_ring, roof_z, roof_index)
        except TriangulationError as e:
            logger.warning(
                f"Building {building.osm_id}: Triangulation failed: {e}"
            )
            building.warnings.append(f"Flat roof: triangulation failed: {e}")
            # Last resort: try a fan triangulation from centroid
            _generate_fan_roof(mesh, footprint.outer_ring, roof_z, roof_index)

    return mesh


def _generate_roof_with_holes(
    mesh: MeshData,
    footprint: Polygon,
    roof_z: float,
    roof_index: int = 0
) -> None:
    """
    Generate triangulated roof surface with holes and UV coordinates.

    Args:
        mesh: MeshData to add to
        footprint: Polygon with outer ring and holes
        roof_z: Roof elevation
        roof_index: Roof pattern index [0..5] for UV mapping
    """
    # Triangulate with hole bridging
    merged_vertices, triangles = triangulate_with_holes(
        footprint.outer_ring,
        footprint.holes
    )

    # Get V range for this roof pattern
    v_min, v_max = get_roof_v_range(roof_index)

    # Add vertices and UVs to mesh
    # UV mapping: use world XY coordinates scaled to tile the texture
    # 3m = 1.0 UV unit for consistency with wall textures
    vertex_indices = []
    uv_indices = []
    for point in merged_vertices:
        v_idx = mesh.add_vertex(point.x, point.y, roof_z)
        vertex_indices.append(v_idx)

        # Compute UV based on world position
        # U: horizontal tiling (can wrap)
        # V: map to center of roof slice to avoid edge bleeding
        u = point.x / 3.0  # 3m = 1.0 UV
        v = (v_min + v_max) / 2.0  # Center of slice
        uv_idx = mesh.add_uv(u, v)
        uv_indices.append(uv_idx)

    # Add triangles with UVs
    for a, b, c in triangles:
        mesh.add_triangle_with_uvs(
            vertex_indices[a], vertex_indices[b], vertex_indices[c],
            uv_indices[a], uv_indices[b], uv_indices[c]
        )


def _generate_simple_roof(
    mesh: MeshData,
    ring: List[Point2D],
    roof_z: float,
    roof_index: int = 0
) -> None:
    """
    Generate triangulated roof surface for simple polygon (no holes) with UVs.

    Args:
        mesh: MeshData to add to
        ring: Outer ring vertices
        roof_z: Roof elevation
        roof_index: Roof pattern index [0..5] for UV mapping
    """
    # Skip closing vertex if present
    n = len(ring)
    if n > 0 and ring[0].x == ring[-1].x and ring[0].y == ring[-1].y:
        ring = ring[:-1]
        n -= 1

    if n < 3:
        return

    # Triangulate
    triangles = triangulate_polygon(ring)

    # Get V range for this roof pattern
    v_min, v_max = get_roof_v_range(roof_index)

    # Add vertices and UVs to mesh
    vertex_indices = []
    uv_indices = []
    for point in ring:
        v_idx = mesh.add_vertex(point.x, point.y, roof_z)
        vertex_indices.append(v_idx)

        # Compute UV based on world position
        u = point.x / 3.0  # 3m = 1.0 UV
        v = (v_min + v_max) / 2.0  # Center of slice
        uv_idx = mesh.add_uv(u, v)
        uv_indices.append(uv_idx)

    # Add triangles with UVs
    for a, b, c in triangles:
        mesh.add_triangle_with_uvs(
            vertex_indices[a], vertex_indices[b], vertex_indices[c],
            uv_indices[a], uv_indices[b], uv_indices[c]
        )


def _generate_fan_roof(
    mesh: MeshData,
    ring: List[Point2D],
    roof_z: float,
    roof_index: int = 0
) -> None:
    """
    Generate fan triangulation from centroid with UVs (fallback).

    This works for convex polygons and is a last resort for
    when ear clipping fails.

    Args:
        mesh: MeshData to add to
        ring: Outer ring vertices
        roof_z: Roof elevation
        roof_index: Roof pattern index [0..5] for UV mapping
    """
    # Skip closing vertex if present
    n = len(ring)
    if n > 0 and ring[0].x == ring[-1].x and ring[0].y == ring[-1].y:
        ring = ring[:-1]
        n -= 1

    if n < 3:
        return

    # Get V range for this roof pattern
    v_min, v_max = get_roof_v_range(roof_index)
    v_center = (v_min + v_max) / 2.0

    # Compute centroid
    cx = sum(p.x for p in ring) / n
    cy = sum(p.y for p in ring) / n

    # Add centroid vertex with UV
    center_idx = mesh.add_vertex(cx, cy, roof_z)
    center_uv_idx = mesh.add_uv(cx / 3.0, v_center)

    # Add ring vertices with UVs
    vertex_indices = []
    uv_indices = []
    for point in ring:
        v_idx = mesh.add_vertex(point.x, point.y, roof_z)
        vertex_indices.append(v_idx)

        uv_idx = mesh.add_uv(point.x / 3.0, v_center)
        uv_indices.append(uv_idx)

    # Create fan triangles with UVs
    for i in range(n):
        j = (i + 1) % n
        mesh.add_triangle_with_uvs(
            center_idx, vertex_indices[i], vertex_indices[j],
            center_uv_idx, uv_indices[i], uv_indices[j]
        )


def generate_flat_cap(
    ring: List[Point2D],
    z: float,
    osm_id: Optional[str] = None,
    facing_up: bool = True
) -> MeshData:
    """
    Generate a flat cap surface (for gable end caps, etc.).

    Args:
        ring: Ring vertices
        z: Z elevation
        osm_id: Optional building ID
        facing_up: If True, normals face +Z; if False, face -Z

    Returns:
        MeshData with cap geometry
    """
    mesh = MeshData(osm_id=osm_id)

    # Skip closing vertex if present
    n = len(ring)
    if n > 0 and ring[0].x == ring[-1].x and ring[0].y == ring[-1].y:
        ring = ring[:-1]
        n -= 1

    if n < 3:
        return mesh

    try:
        triangles = triangulate_polygon(ring)
    except TriangulationError:
        # Fallback to fan
        _generate_fan_roof(mesh, ring, z)
        return mesh

    # Add vertices
    vertex_indices = []
    for point in ring:
        idx = mesh.add_vertex(point.x, point.y, z)
        vertex_indices.append(idx)

    # Add triangles (reverse order if facing down)
    for a, b, c in triangles:
        if facing_up:
            mesh.add_triangle(
                vertex_indices[a],
                vertex_indices[b],
                vertex_indices[c]
            )
        else:
            mesh.add_triangle(
                vertex_indices[c],
                vertex_indices[b],
                vertex_indices[a]
            )

    return mesh
